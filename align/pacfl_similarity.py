"""P2: private pairwise PACFL similarity between two clients -> XOR-shared "similar" bit.

  s_ij = ||U_i^T U_j||_F^2 / min(p_i, p_j) = <P_i, P_j>_F / min(p_i, p_j),   P = U U^T (r x r)
(U: orthonormal basis of the client's data after the public projection, align.pacfl_plain.)

1. OLE (Gilboa over IKNP word OT). The lower-index client i is the OT chooser with
   a = upper-triangle(P_i) in fixed point, shifted to be nonnegative; j is the sender with
   b = weighted upper-triangle(P_j) (off-diagonal x2). For each bit t of a_k, j offers
   (r, r + 2^t b_k). Result: additive shares z_i + z_j = <P_i, P_j> 2^(2f) mod 2^64.
2. 2PC (GMW, align.mpc with two parties): bit = [z_i + z_j > min(T_i, T_j)], T = tau p 2^(2f),
   so each party's basis size p stays private too. The bit is never opened: i keeps e_i,
   j keeps e_j, e_i ^ e_j = bit. The score itself never exists in the clear.

No server takes part. Cost per pair: (f+2) r(r+1)/2 word OTs + ~300 AND gates, over two IKNP
sessions (one per direction) whose base OTs dominate: ~2.5 s per pair in pure Python.
ponytail: n(n-1)/2 pairs; 30 clients ~ 18 CPU-min, spread over processes (workers).
"""
import secrets
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from secfl.ot import IKNPBitSession
from .circuits import add, lt, cond_swap, to_bits
from .mpc import MPC

FRAC = 12            # fixed-point bits of P entries (|P_kl| <= 1)
WIDTH = 40           # comparison width: <P_i,P_j> 2^24 <= p 2^24 < 2^33 for p <= 256
BIAS = 1 << 34       # added to both sides so fixed-point noise around 0 cannot wrap


def pair_vectors(U, frac=FRAC):
    """(chooser input a >= 0 with frac+2 bits, sender input b as uint64) for one basis."""
    P = U @ U.T
    iu = np.triu_indices(len(P))
    p = P[iu]
    a = (np.rint(p * 2 ** frac).astype(np.int64) + (1 << frac)).astype(np.uint64)    # in [0, 2^(f+1)]
    b = (np.rint(p * np.where(iu[0] == iu[1], 1., 2.) * 2 ** frac).astype(np.int64)).astype(np.uint64)
    return a, b


def _rand_u64(shape):
    n = int(np.prod(shape))
    return np.frombuffer(secrets.token_bytes(8 * n), np.uint64).reshape(shape).copy()


def ole_inner(a, b, ot: IKNPBitSession, nbits):
    """Gilboa: chooser has a (nbits each), sender has b; returns shares (z_chooser, z_sender)
    with z_chooser + z_sender = sum_k a_k b_k mod 2^64."""
    t = np.arange(nbits, dtype=np.uint64)
    bits = ((a[:, None] >> t) & np.uint64(1)).astype(np.uint8)          # chooser's choices
    r = _rand_u64(bits.shape)                                            # sender's masks
    got = ot.transfer_words(r.ravel(), (r + (b[:, None] << t)).ravel(), bits.ravel())
    with np.errstate(over='ignore'):
        return got.sum(dtype=np.uint64), np.uint64(0) - r.sum(dtype=np.uint64)


def threshold(tau, p, frac=FRAC):
    return int(round(tau * p * 2 ** (2 * frac))) + BIAS


def similar_bit(Ui, Uj, tau, session=b'', frac=FRAC):
    """Both clients' view of one pair. Returns (e_i, e_j, stats); e_i ^ e_j = [s_ij > tau]."""
    ai, _ = pair_vectors(Ui, frac)                   # computed by client i
    aj, bj = pair_vectors(Uj, frac)                  # computed by client j
    e = MPC(2, session, chunk=1024)
    zi, zj = ole_inner(ai, bj, e._session(0, 1), frac + 2)    # same i<-j OT session as the 2PC triples
    with np.errstate(over='ignore'):
        zj = zj - (np.uint64(1 << frac) * bj.sum(dtype=np.uint64)) + np.uint64(BIAS)   # j removes the offset
    mask = (1 << WIDTH) - 1
    x = add(e, e.input(to_bits([int(zi) & mask], WIDTH, msb_first=False), 0),
            e.input(to_bits([int(zj) & mask], WIDTH, msb_first=False), 1))[..., ::-1]      # MSB first
    ti = e.input(to_bits([threshold(tau, Ui.shape[1], frac)], WIDTH), 0)
    tj = e.input(to_bits([threshold(tau, Uj.shape[1], frac)], WIDTH), 1)
    smaller, _ = cond_swap(e, lt(e, ti, tj), tj, ti)                   # min(T_i, T_j)
    bit = lt(e, smaller, x)
    return int(bit.parts[0][0]), int(bit.parts[1][0]), dict(word_ots=ai.size * (frac + 2), and_gates=e.and_gates)


def all_pairs(bases, tau, session=b'pacfl', workers=1):
    """Every pair runs similar_bit on its own. Returns E (n x n uint8): client i holds row i,
    and E[i, j] ^ E[j, i] = [s_ij > tau] for i != j. Plus summed cost stats."""
    n = len(bases)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    args = ([bases[i] for i, _ in pairs], [bases[j] for _, j in pairs], [tau] * len(pairs),
            [session + f'/{i}-{j}'.encode() for i, j in pairs])
    if workers > 1:          # base OTs are pure-Python ristretto (~1 s per session): use processes
        with ProcessPoolExecutor(workers) as pool:
            out = list(pool.map(similar_bit, *args))
    else:
        out = list(map(similar_bit, *args))
    E = np.zeros((n, n), np.uint8)
    stats = dict(pairs=len(pairs), word_ots=0, and_gates=0)
    for (i, j), (ei, ej, st) in zip(pairs, out):
        E[i, j], E[j, i] = ei, ej
        stats['word_ots'] += st['word_ots']
        stats['and_gates'] += st['and_gates']
    return E, stats
