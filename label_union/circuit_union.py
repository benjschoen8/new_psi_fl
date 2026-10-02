"""Label union by circuit PSI: the clients group their labels inside an honest-majority MPC, then the
usual two SecAggs give every row its index and KEM keys. One server (the Aggregator); semi-honest.

Public: the encoder + 2000 anchor words (fuzzy keywords), the 45 seeded image anchors
(label_union.domain), thresholds tau (keyword, CSLS) and t (shared image anchors), steps D per check (default 1), rows m.

  A  each client, locally, one row per label: its keyword (exact: the name; fuzzy: unit embedding e
     of the normalized keyword, fixed point, and its CSLS hub term r = mean cosine to its h nearest
     anchor words; a single letter is a 'symbol' that only matches itself, case kept), and its k
     nearest image anchors as a 0/1 vector b; padded with dummy rows to m.
  B  Shamir-share every input among the n clients (degree < n/2); messages relayed by the Aggregator,
     encrypted client to client.
  C  MPC (circuit PSI): for every pair of rows of different clients (a client's own labels are
     never compared directly; they still join one group through other clients' rows)
        match = valid_i valid_j [<b_i, b_j> >= t] [keywords match]
        keywords match = exact: name_i == name_j;  fuzzy: both symbols ? sym_i == sym_j :
                         (either a symbol ? 0 : 2<e_i, e_j> - r_i - r_j >= tau)
     then connected components by min-label propagation: repeat (D steps, open one 'converged'
     bit) until converged; one class is usually a clique, so with D = 1 one step plus one check.
     The first step runs on the public row numbers (smallest matching row = first 1 of the row,
     a prefix-OR), later steps compare secret labels. Each row carries its root's secret kappa. Output: K = kappa of the root, opened ONLY to
     the row's owner. Rows in one group share K; nobody sees the match matrix or the groups.
  D  each client, locally: rows of its own with the same K become one row; bucket b = H(K), sk = H'(K).
  E  SecAgg #1: bucket union over the K's (label_union.oprf_union._union_from_tags) -> index = rank.
  F  SecAgg #2: pk of every index (_pk_secagg).
Leakage: Aggregator U and the pk list; any < n/2 clients: U, their own outputs, the 'converged'
bits (= how many propagation steps the longest chain of matches needed, usually 1).

ponytail: step C runs here as its ideal functionality (the exact integer computation the MPC
evaluates, on the same fixed-point inputs), so accuracy and indices are what the MPC would produce;
its cost is the operation count of mpc_cost (Shamir/DN07), to be measured with MP-SPDZ for the paper.
"""
import functools
import secrets
import time

import numpy as np

from label_union.oprf_union import _union_from_tags, _pk_secagg, slot_key, canonical, BUCKET_BITS

FIX = 7                                   # fixed point: e * 2^7, dot products and r at 2^14 (CSLS
                                          # resolution 6e-5; real-data groupings same as at 2^12)
TAU = 0.10                                # CSLS threshold (CIFAR names: synonyms >= .21, others <= .01)
HUB = 5                                   # CSLS: h nearest anchor words
ANCHORS = 2000


def _symbol(t):
    return len(t) == 1 and t.isascii() and t.isalpha()


@functools.lru_cache(maxsize=4)
def hub_anchors(model):
    from label_union import encoder, fuzzy_union as fz
    return encoder.embed(fz.anchor_words(ANCHORS), model)


def keyword_rows(texts, fuzzy, model=None):
    """texts: one client's {label: keyword}. Exact: {label: ('name', bytes)}; fuzzy: symbols
    ('sym', letter) or ('emb', int vector, int r)."""
    if not fuzzy:
        return {x: ('name', canonical(t)) for x, t in texts.items()}
    from label_union import encoder, fuzzy_union as fz
    norm = {x: fz.normalize(t) for x, t in texts.items()}
    words = [x for x, t in norm.items() if not _symbol(t)]
    out = {x: ('sym', t) for x, t in norm.items() if _symbol(t)}
    if words:
        model = model or encoder.DEFAULT_MODEL
        E = encoder.embed([norm[x] for x in words], model)
        r = np.sort(E @ hub_anchors(model).T, 1)[:, -HUB:].mean(1)
        for x, e, ri in zip(words, E, r):
            out[x] = ('emb', np.round(e * (1 << FIX)).astype(np.int64), int(round(ri * (1 << 2 * FIX))))
    return out


def image_rows(sets, n_anchors=45):
    """{label: tuple of k anchor ids} -> {label: 0/1 int vector}."""
    out = {}
    for x, s in sets.items():
        v = np.zeros(n_anchors, np.int64)
        v[list(s)] = 1
        out[x] = v
    return out


def _kw_match(a, b, tau):
    if a[0] == 'name' or b[0] == 'name':
        return a == b
    if a[0] == 'sym' or b[0] == 'sym':
        return a[0] == b[0] == 'sym' and a[1] == b[1]
    return 2 * int(a[1] @ b[1]) - a[2] - b[2] >= int(round(tau * (1 << 2 * FIX)))


def group(rows, tau=TAU, t=2, owners=None):
    """Ideal functionality of step C on the real rows (dummies never match). rows: list of
    (kw, img or None); owners: client of each row (rows of one client are not compared).
    Returns (group id per row, propagation steps the components need)."""
    N = len(rows)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if owners is not None and owners[i] == owners[j]:
                continue
            (ki, bi), (kj, bj) = rows[i], rows[j]
            if (bi is None or int(bi @ bj) >= t) and _kw_match(ki, kj, tau):
                adj[i].append(j), adj[j].append(i)
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    root = [find(a) for a in range(N)]
    steps = 0                                  # min-label propagation: steps until the minimum
    for r in set(root):                        # reaches every member = eccentricity of the root
        dist, frontier, seen = 0, [r], {r}
        while frontier:
            nxt = [b for a in frontier for b in adj[a] if b not in seen]
            seen.update(nxt)
            frontier = nxt
            dist += bool(nxt)
        steps = max(steps, dist)
    return root, steps


def mpc_cost(n, m, dim, n_img, D, steps, field_bytes=8):
    """Operation count of step C for N = n*m padded rows (Shamir, honest majority, DN07 degree
    reduction: ~2 field elements sent per party per multiplication incl. preprocessing).
    Propagation: ceil(steps / D) repeats of D steps + 1 check (a check costs a step); the first
    step uses the public row numbers (prefix-OR + selection, ~2N^2)."""
    N = n * m
    pairs = N * (N - 1) // 2 - n * m * (m - 1) // 2               # rows of one client: not compared
    cmp_kw, cmp_small, eq = 2 * (2 * FIX + 18), 8, 2 * 32       # bit-decomposition comparisons
    per_pair = 2 + cmp_kw + cmp_small + eq + 6                    # 2 dot products (1 reshare each)
    repeats = max(1, -(-steps // D))
    iters = repeats * (D + 1)                                     # steps + checks
    lr = max(1, int(np.ceil(np.log2(N + 1))))
    per_iter = 4 * N * N + N * (N - 1) * 2 * (lr + 1)            # selects (root + 3-word kappa), mins
    mults = pairs * per_pair + 2 * N * N + (iters - 1) * per_iter
    inputs = m * (dim + n_img + 3) * (n - 1)                     # shares sent per client
    return dict(rows=N, pairs=pairs, mults=int(mults), propagation_steps=int(iters), repeats=int(repeats),
                rounds=int(12 + iters * (lr + 2 * lr)),
                bytes_per_client=int((inputs + 2 * mults) * field_bytes))


def circuit_union_with_keys(client_labels, client_keywords, client_sets=None, fuzzy=False, tau=TAU, t=2,
                            D=1, m=None, workers=1, bucket_bits=BUCKET_BITS, session=b'label-union-circuit',
                            secure=True):
    """client_labels: per client [labels]; client_keywords: per client {label: keyword} (exact: the name);
    client_sets: per client {label: k nearest image anchors} or None (no image check).
    secure=False: the same grouping in the clear (Plain-GeFL): no MPC, no SecAgg.
    Returns (index per client {label: row}, sks per client {label: sk} or None, pks or None, U, stats)."""
    t0 = time.perf_counter()
    kws = [keyword_rows(k, fuzzy) for k in client_keywords]                       # A (local)
    imgs = [image_rows(s) for s in client_sets] if client_sets else None
    owner, rows = [], []
    for c, labels in enumerate(client_labels):
        for x in labels:
            owner.append((c, x))
            rows.append((kws[c][x], imgs[c][x] if imgs else None))
    root, steps = group(rows, tau, t, [c for c, _ in owner])                      # C (ideal functionality)
    kappa = {r: secrets.token_bytes(16) for r in set(root)}
    K = [dict() for _ in client_labels]
    for (c, x), r in zip(owner, root):
        K[c][x] = kappa[r]                                                        # opened to the owner only
    m = m or max(len(l) for l in client_labels)
    dim = 384 if fuzzy else 1
    cost = mpc_cost(len(client_labels), m, dim, 45 if imgs else 0, D, steps)
    group_seconds = time.perf_counter() - t0
    if not secure:
        order = {r: g for g, r in enumerate(sorted(set(root)))}
        index = [{x: order[r] for (c2, x), r in zip(owner, root) if c2 == c} for c in range(len(client_labels))]
        return index, None, None, len(order), dict(method='plain-circuit', groups=len(order),
                                                   setup_upload_bytes_per_client=0, setup_download_bytes_per_client=0)
    tags = [{v: v for v in set(k.values())} for k in K]                          # D: one row per distinct K
    idx, U, stats = _union_from_tags(tags, workers, bucket_bits, session + b'/union')   # E
    sks = [{v: slot_key(v) for v in own} for own in tags]
    pks = _pk_secagg([{idx[c][v]: sk for v, sk in own.items()} for c, own in enumerate(sks)], U,
                     session + b'/pk', workers, stats['secagg'])                 # F
    index = [{x: idx[c][v] for x, v in K[c].items()} for c in range(len(K))]
    sk_out = [{x: sks[c][v] for x, v in K[c].items()} for c in range(len(K))]
    sa, n = stats['secagg'], len(client_labels)
    stats.update(method='circuit-psi', mpc=dict(cost, estimated=True, tau=tau, t=t, D=D),
                 group_seconds=group_seconds, seconds=time.perf_counter() - t0,
                 setup_upload_bytes_per_client=cost['bytes_per_client'] + (sa['payload_up'] + sa['control_up']) / n,
                 setup_download_bytes_per_client=cost['bytes_per_client'] + sa['control_down'] / n + U * (4 + 32))
    return index, sk_out, pks, U, stats
