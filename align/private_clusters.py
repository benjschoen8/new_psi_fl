"""P3: private PACFL groups from the pairwise shared bits of P2: n-party MPC among the clients only.

Input:  E (n x n) from align.pacfl_similarity.all_pairs: client i holds row i and
        A[i, j] = E[i, j] ^ E[j, i] = [s_ij > tau];  each client's dictionary indicator.
Circuit (align.mpc GMW, every value XOR-shared among all n clients):
  1. A -> reflexive transitive closure R by ceil(log2 n) boolean squarings
     (R[u, v] = 1 iff u and v are in the same connected component = single-linkage group)
  2. leader l_v = [no u < v with R[u, v]]  (the smallest index of each group)
  3. rank = exclusive prefix sum of l;  G = number of groups (opened: the server needs the
     layout and sees the groups anyway)
  4. slot_v = XOR_u (R[u, v] & l_u) & rank_u   (exactly one leader per group)
  5. labels[s, d] = XOR_u [rank_u = s & l_u] & OR_v (R[u, v] & ind_v[d])   (group label sets)
Outputs:
  client v:  its row R[v, :] (who it is grouped with) and its slot in [0, G)
  everyone:  G
  server:    labels (G x |dictionary|) existence bits per group; never members, sizes or counts.
The server takes no part in the computation, only receives the opened `labels`.

ponytail: ~2 n^3 ceil(log2 n) AND gates for the closure; 30 clients ~ 300k ANDs, i.e. ~5
triple batches over 870 IKNP sessions: minutes in pure Python, once, before training.
"""
import math

import numpy as np

from .circuits import and_reduce, eq, prefix_sum, to_bits, from_bits
from .mpc import MPC, Shared, concat


def _local_and(x: Shared, public_bits) -> Shared:
    return Shared([p & np.asarray(public_bits, np.uint8) for p in x.parts])


def _T(x: Shared) -> Shared:
    return Shared([p.T for p in x.parts])


def _or_reduce(e, x: Shared) -> Shared:
    return ~and_reduce(e, ~x)


def private_clusters(E, indicators, session=b'private-pacfl', workers=1):
    E = np.asarray(E, np.uint8)
    n = len(E)
    ind = np.asarray(indicators, np.uint8)
    e = MPC(n, session).warm(workers)
    # A: symmetric, diagonal 1. Pair (i, j) contributes both shares; nobody sees A.
    zero = np.zeros((n, n), np.uint8)
    A = e.constant(np.eye(n, dtype=np.uint8))
    for i in range(n):
        own = zero.copy()
        own[i, :] = E[i]
        own[:, i] = E[i]
        own[i, i] = 0
        A = A ^ e.input(own, i)
    # 1. closure by squaring: R[i, j] <- OR_k R[i, k] & R[k, j]
    R = A
    for _ in range(max(1, math.ceil(math.log2(max(2, n - 1))))):
        left = Shared([np.broadcast_to(p[:, None, :], (n, n, n)) for p in R.parts])      # [i, j, k] = R[i, k]
        right = Shared([np.broadcast_to(p.T[None, :, :], (n, n, n)) for p in R.parts])   # [i, j, k] = R[k, j]
        R = _or_reduce(e, e.and_(left, right))
    # 2. leaders: AND_{u < v} ~R[u, v]
    below = np.triu(np.ones((n, n), np.uint8), 1)                       # [u, v] = u < v
    lead = and_reduce(e, _T(~_local_and(R, below)))                     # over u
    # 3. ranks (LSB-first words): exclusive prefix sum; G = total
    w = n.bit_length() + 1
    words = concat([e.constant(np.zeros((n + 1, 1), np.uint8)), Shared([np.zeros((n + 1, w - 1), np.uint8)] * n)])
    words = words.set((slice(1, None), 0), lead)
    ranks = prefix_sum(e, words)                                        # ranks[u] = #leaders < u
    G = int(from_bits(e.open(ranks[n]), msb_first=False))
    rank = ranks[:n]
    # 4. slots
    LR = e.and_(R, lead.reshape(n, 1).broadcast_to((n, n)))             # [u, v] = R[u, v] & l_u
    slot = e.and_(_T(LR).reshape(n, n, 1).broadcast_to((n, n, w)),
                  rank.reshape(1, n, w).broadcast_to((n, n, w))).xor_reduce(1)                 # [v, bit]
    # 5. group label sets for the server, ordered by slot
    D = ind.shape[1]
    I = concat([e.input(ind[v].reshape(1, D), v) for v in range(n)], axis=0)                    # [v, d]
    has = _or_reduce(e, e.and_(Shared([np.broadcast_to(p[:, None, :], (n, D, n)) for p in R.parts]),     # [u, d, v] = R[u, v]
                               Shared([np.broadcast_to(p.T[None, :, :], (n, D, n)) for p in I.parts])))  # [u, d, v] = I[v, d]
    sel = e.and_(eq(e, rank.reshape(1, n, w).broadcast_to((G, n, w)),
                    e.constant(np.broadcast_to(to_bits(np.arange(G), w, msb_first=False)[:, None, :], (G, n, w)))),
                 lead.reshape(1, n).broadcast_to((G, n)))                                       # [s, u]
    labels = e.and_(sel.reshape(G, n, 1).broadcast_to((G, n, D)),
                    has.reshape(1, n, D).broadcast_to((G, n, D))).xor_reduce(1)                  # [s, d]
    clients = [dict(members=[int(u) for u in np.flatnonzero(e.reveal_to(R[v], v))],
                    slot=int(from_bits(e.reveal_to(slot[v], v), msb_first=False))) for v in range(n)]
    return dict(clients=clients, groups=G, labels=e.open(labels).astype(bool),
                stats=dict(and_gates=e.and_gates, rounds=e.rounds, parties=n))


def plain_reference(A, indicators):
    """Same outputs computed in the clear (tests / experimenter only)."""
    from .pacfl_plain import components
    return from_groups(components(np.asarray(A, bool) | np.eye(len(A), dtype=bool)), indicators)


def from_groups(groups, indicators):
    """Outputs of private_clusters for a given grouping (plaintext baselines)."""
    groups = sorted(sorted(int(v) for v in g) for g in groups)          # slot order = smallest member
    slot = {v: s for s, g in enumerate(groups) for v in g}
    ind = np.asarray(indicators, bool)
    return dict(clients=[dict(members=groups[slot[v]], slot=slot[v]) for v in range(len(ind))],
                groups=len(groups), labels=np.array([ind[g].any(0) for g in groups]))
