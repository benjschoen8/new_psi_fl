"""Label union by multi-party OPRF (PSI-style) + one SecAgg. No dictionary, no committee, no
Aggregator in the tagging step. Semi-honest.

1. keys     client i draws a key share k_i; the joint key K = prod k_i exists nowhere.
2. tags     for each own label x, client i blinds P = H(x)*r and sends it around the ring of
            clients; every client multiplies by its k_j; i applies k_i and removes r:  T(x) = H(x)*K.
            Equal labels -> equal tags; a tag needs all n key shares. Every hop sees only
            uniformly random group elements (DDH), and every list is padded to m_max, so
            nobody learns how many labels another client has.
3. union    client i sets a random nonzero value in bucket b(T) = SHA-256(T) mod B of a
            B-entry vector; SecAgg reveals only the sum. Occupied buckets = the union; a sum of
            random nonzero values is random, so holder counts stay hidden.
4. indices  the Aggregator publishes its sorted occupied buckets: its label table is just the
            index list 0..U-1. Client i's index for x = position of b(T(x)) in that list.
            Buckets are pseudorandom, so an index says nothing about which labels exist.

Leakage: Aggregator: U only. Clients: U. Nobody: who holds what, names, counts.
Collisions (two labels in one bucket, prob ~ U^2 / 2B per try) are detected by a purity check
and the SecAgg is redone with a new public salt: no silent merges (see oprf_union). ponytail: pure-Python ristretto, ~2.5 ms per multiplication: n^2 m_max
mults in total (30 clients x 62 labels ~ 2.5 CPU-min); libsodium for real deployments.

Weak image check: with domains (label_union.domain) the OPRF input is name | domain code, where the
code says what kind of picture the label's samples are (strokes vs photo ...), computed locally
against synthetic public anchors. A digit "cat" and a photo "cat" then get different indices.
"""
import hashlib
import itertools
import secrets
import struct
import time
import unicodedata

import numpy as np

from secfl import ristretto as rg
from secfl.secagg import run_secagg

DST = b'label-union-oprf-v1'
BUCKET_BITS = 20


def canonical(label) -> bytes:
    """Label -> bytes. NFKC only: case is kept (EMNIST 'A' and 'a' are different classes)."""
    return unicodedata.normalize('NFKC', str(label)).strip().encode('utf-8')


def _ring_pass(points, keys, owner):
    """Every client except the owner multiplies the owner's list by its key share, in ring order."""
    n = len(keys)
    for step in range(1, n):
        j = (owner + step) % n
        points = [p * keys[j] for p in points]
    return points


def tags(client_labels, m_max=None, workers=1, domains=None):
    """Step 2 for all clients. Returns per client {label: tag bytes}.
    domains: optional per client {label: domain code} (label_union.domain); the OPRF input is then
    name | code, so equal names from different kinds of pictures get different tags."""
    n = len(client_labels)
    m_max = m_max or max(len(l) for l in client_labels)
    if any(len(set(map(canonical, l))) != len(l) or len(l) > m_max for l in client_labels):
        raise ValueError('each client needs <= m_max distinct labels')
    keys = [rg.random_scalar() for _ in range(n)]                      # client i keeps keys[i]
    jobs = []
    for i, labels in enumerate(client_labels):
        r = [rg.random_scalar() for _ in range(m_max)]
        real = [rg.hash_to_group(canonical(x) + (b'\x00' + domains[i][x].encode() if domains else b''), DST)
                for x in labels]
        pad = [rg.BASE * rg.random_scalar() for _ in range(m_max - len(labels))]
        jobs.append((i, r, [p * s for p, s in zip(real + pad, r)]))  # blinded, padded
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            passed = list(pool.map(_ring_pass, [b for _, _, b in jobs], [keys] * n, [i for i, _, _ in jobs]))
    else:
        passed = [_ring_pass(b, keys, i) for i, _, b in jobs]
    out = []
    for (i, r, _), pts, labels in zip(jobs, passed, client_labels):
        final = [p * ((keys[i] * pow(s, -1, rg.L)) % rg.L) for p, s in zip(pts, r)]   # own share, unblind
        out.append({x: final[a].encode() for a, x in enumerate(labels)})
    return out


def bucket(tag: bytes, bits=BUCKET_BITS, salt=0) -> int:
    return int.from_bytes(hashlib.sha256(b'bucket/%d/' % salt + tag).digest()[:8], 'big') >> (64 - bits)


def check(tag: bytes, salt=0) -> np.uint64:
    """Odd 64-bit fingerprint for the purity check."""
    return np.uint64(int.from_bytes(hashlib.sha256(b'check/%d/' % salt + tag).digest()[:8], 'big') | 1)


def _upload(own, bits, salt):
    """Client vector [A | B]: bucket b(T) gets A = r, B = r * check(T) (mod 2^64), r uniform mod 2^64.
    Uniform (not forced odd): a sum of k uniform words is uniform, so A hides the holder count,
    parity included."""
    b = [bucket(v, bits, salt) for v in own.values()]
    vec = np.zeros(2 << bits, np.uint64)
    if len(set(b)) != len(b):
        return None                                                     # own labels collide: retry
    r = np.frombuffer(secrets.token_bytes(8 * len(b)), np.uint64)
    with np.errstate(over='ignore'):
        vec[b] = r
        vec[np.array(b, dtype=np.int64) + (1 << bits)] = r * np.array([check(v, salt) for v in own.values()])
    return vec


def oprf_union(client_labels, workers=1, bucket_bits=BUCKET_BITS, m_max=None, session=b'label-union', max_tries=20,
               domains=None):
    """Returns (index per client {label: index}, U, stats).

    Collisions are detected, never silently merged: a bucket holding one label T satisfies
    B = check(T) * A (mod 2^64); with two labels this fails except with prob ~2^-60. Every
    client checks its own buckets; any failure (or two own labels in one bucket) makes all
    clients redo only the SecAgg with the next public salt (the tags are reused). That
    reveals only that a collision happened (prob ~ U^2 / 2^(bits+1) per try)."""
    t = time.perf_counter()
    T = tags(client_labels, m_max, workers, domains)
    tag_seconds = time.perf_counter() - t
    index, U, stats = _union_from_tags(T, workers, bucket_bits, session, max_tries)
    stats.update(tag_seconds=tag_seconds, seconds=time.perf_counter() - t,
                 multiplications=len(T) ** 2 * max(len(l) for l in client_labels))
    return index, U, stats


def _union_from_tags(T, workers, bucket_bits, session, max_tries=20):
    n = len(T)
    sa = {}
    for salt in range(max_tries):
        vectors = {i: _upload(own, bucket_bits, salt) for i, own in enumerate(T)}
        if any(v is None for v in vectors.values()):
            continue
        total, _ = run_secagg(vectors, threshold=max(2, -(-2 * n // 3)), modulus_bits=64,
                              session=session + b'/%d' % salt, workers=workers, stats=sa)
        A, B = total[:1 << bucket_bits], total[1 << bucket_bits:]
        occupied = np.flatnonzero(A | B)                                # Aggregator: sorted buckets
        with np.errstate(over='ignore'):
            clean = all(A[bucket(v, bucket_bits, salt)] != 0 and              # A = 0: prob 2^-64, retry
                        B[bucket(v, bucket_bits, salt)] == check(v, salt) * A[bucket(v, bucket_bits, salt)]
                        for own in T for v in own.values())            # each client checks its own
        if clean:
            break
    else:
        raise RuntimeError(f'bucket collisions in {max_tries} tries; raise bucket_bits')
    rank = {int(b): k for k, b in enumerate(occupied)}                  # published list -> indices
    index = [{x: rank[bucket(v, bucket_bits, salt)] for x, v in own.items()} for own in T]
    return index, len(occupied), dict(bucket_bits=bucket_bits, tries=salt + 1,
                                      upload_bytes_per_client=16 << bucket_bits, secagg=sa)


def slot_key(tag: bytes) -> int:
    """KEM secret key of a union index, from its tag: only holders of the label can compute it."""
    return rg.hash_to_scalar(tag, b'label-union-sk-v1')


PK_CHUNKS = 16                          # a 32-byte pk as 16 x 16-bit chunks


def _pk_secagg(entries, U, session, workers, sa):
    """entries: per client {slot: sk} (every holder of a slot has the same sk). One SecAgg of U x 17
    words gives the Aggregator pk_0..pk_{U-1} and nothing else: every holder of slot k writes r and
    r*c_j (mod 2^64) for the 16 16-bit chunks c_j of pk_k, r uniform; A = sum r = 2^s a' (a' odd) gives
    c_j = (B_j / 2^s) a'^-1 mod 2^(64-s), exact while s <= 48 (else redo, prob 2^-48). A is uniform
    whatever the number of holders, so it hides the count. Non-holders write zeros."""
    n = len(entries)
    chunks = [{k: [int(c) for c in np.frombuffer((rg.BASE * sk).encode(), '>u2')] for k, sk in own.items()}
              for own in entries]
    for attempt in range(4):
        vectors = {}
        for i, own in enumerate(chunks):
            v = np.zeros((U, PK_CHUNKS + 1), np.uint64)
            for k, cs in own.items():
                r = secrets.randbits(64)
                v[k] = [r] + [(r * c) % (1 << 64) for c in cs]
            vectors[i] = v.ravel()
        total, _ = run_secagg(vectors, threshold=max(2, -(-2 * n // 3)), modulus_bits=64,
                              session=session + b'/%d' % attempt, workers=workers, stats=sa)
        pks = [decode_pk([int(w) for w in row]) for row in total.reshape(U, PK_CHUNKS + 1)]
        if all(p is not None for p in pks):
            return pks
    raise RuntimeError('pk aggregation failed 4 times: holders disagree on a pk')


def decode_pk(row):
    """[A, B_1..B_16] (mod 2^64 sums) -> 32-byte pk, or None if A has >48 factors of 2 (redo)."""
    A = row[0]
    if A == 0:
        return None
    s = (A & -A).bit_length() - 1
    if s > 48:
        return None
    mod = 1 << (64 - s)
    inv = pow(A >> s, -1, mod)
    out = []
    for B in row[1:]:
        if B % (1 << s):
            return None                                                 # holders disagree
        c = (B >> s) * inv % mod
        if c >= 1 << 16 or (c * A - B) % (1 << 64):
            return None
        out.append(c)
    return np.array(out, '>u2').tobytes()


def oprf_union_with_keys(client_labels, workers=1, bucket_bits=BUCKET_BITS, domains=None, session=b'label-union'):
    """oprf_union + one KEM key pair per index (for secure_main's per-label downlink).

    Returns (slots, keys, pks, stats) like secure_main.setup_union:
      slots  per client {label: index in [0, U)}      keys  per client {label: sk}
      pks    pk of every index (the Aggregator's view, posted on the BB)
    sk = slot_key(tag); pk = sk*B. The Aggregator gets pk_k without learning who sent it: a second
    SecAgg of U x 17 words where every holder of k writes r and r*c_j (mod 2^64) for the 16 16-bit
    chunks c_j of pk_k (r uniform mod 2^64, same pk for all holders). With A = sum r = 2^s a' (a' odd):
    c_j = (B_j / 2^s) a'^-1 mod 2^(64-s), exact while s <= 48 (else redo, prob 2^-48). A is uniform
    whatever the number of holders, so it hides the count. Non-holders write zeros."""
    t = time.perf_counter()
    T = tags(client_labels, None, workers, domains)
    index, U, stats = _union_from_tags(T, workers, bucket_bits, session)
    keys = [{x: slot_key(v) for x, v in own.items()} for own in T]
    pks = _pk_secagg([{idx[x]: sk for x, sk in own.items()} for idx, own in zip(index, keys)], U,
                     session + b'/pk', workers, stats['secagg'])
    n, m = len(T), max(len(l) for l in client_labels)
    sa = stats['secagg']
    stats.update(seconds=time.perf_counter() - t, pk_upload_bytes_per_client=8 * U * (PK_CHUNKS + 1),
                 # every padded list of m points (32 B) makes n hops: n*m*32 bytes sent per client
                 ring_bytes_per_client=n * m * 32,
                 setup_upload_bytes_per_client=n * m * 32 + (sa['payload_up'] + sa['control_up']) / n,
                 setup_download_bytes_per_client=n * m * 32 + sa['control_down'] / n + U * (4 + 32))   # + index list, pks
    return index, keys, pks, stats


# ------------------------------------------------------------------ t-out-of-k image matching
IMG = '\x00img:'


def subsets(anchors, t=2):
    """The t-subsets of a label's k nearest image anchors (label_union.domain.anchor_set), as strings."""
    return [','.join(map(str, c)) for c in itertools.combinations(sorted(anchors), t)]


def _seal(pk: bytes, info: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from secfl.kem import encaps
    enc, k = encaps(pk, info)
    nonce = secrets.token_bytes(12)
    return enc + nonce + AESGCM(k).encrypt(nonce, data, info)


def _open(sk: int, pk: bytes, info: bytes, blob: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from secfl.kem import decaps
    return AESGCM(decaps(sk, blob[:32], pk, info)).decrypt(blob[32:44], blob[44:], info)


def components(M, i, j):
    """Connected components of a graph on 0..M-1 with edges (i[e], j[e]) -> group id per vertex,
    groups numbered by their smallest vertex."""
    parent = list(range(M))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(i, j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(a) for a in range(M)]
    order = {r: g for g, r in enumerate(sorted(set(roots)))}
    return [order[r] for r in roots]


def plain_pairs_grouping(client_keys, client_sets, t=2):
    """Plain-GeFL / ground truth: the same grouping in the clear. Two labels share a row iff they are
    joined by a chain of labels with the same key whose anchor sets share >= t anchors.
    Returns (index per client {label: group}, U)."""
    items = sorted({(kw, s) for keys, sets in zip(client_keys, client_sets) for x, kw in keys.items()
                    for s in subsets(sets[x], t)})
    pos = {v: p for p, v in enumerate(items)}
    i, j = [], []
    for keys, sets in zip(client_keys, client_sets):
        for kw in set(keys.values()):
            ps = sorted({pos[kw, s] for x, k in keys.items() if k == kw for s in subsets(sets[x], t)})
            i += [ps[0]] * (len(ps) - 1)
            j += ps[1:]
    group = components(len(items), i, j)
    index = [{x: group[pos[kw, subsets(sets[x], t)[0]]] for x, kw in keys.items()}
             for keys, sets in zip(client_keys, client_sets)]
    return index, max(group) + 1


def pairs_union_with_keys(client_keys, client_sets, t=2, workers=1, bucket_bits=BUCKET_BITS,
                          session=b'label-union-pairs'):
    """Label union with t-out-of-k image matching (fuzzy on the image side), one row + KEM key per group.

    client_keys  per client {label: key}   (exact: the name; fuzzy: its anchor key; labels of one client
                                            with the same key share a row)
    client_sets  per client {label: its k nearest public image anchors} (label_union.domain.client_sets)
    Two labels share a row iff their keys are equal and their anchor sets share >= t anchors, closed
    under chains (A~B, B~C -> one row): the relation is not transitive, so the Aggregator joins the
    matches into connected groups without learning names, holders or holder counts:

    1. ring OPRF   tags of the key (T_kw) and of every t-subset s of the anchor set (T_s = OPRF(key|s))
    2. SecAgg #1   bucket union of the subset tags (as oprf_union): M occupied buckets, published;
                   each client knows the positions of its own subset tags
    3. SecAgg #2   edges: every label writes a random word on (first, p) for each other position p of
                   its subset tags (upper triangle of an M x M matrix). The sum shows the Aggregator
                   which buckets lie in one label, never by whom or how many times
    4. Aggregator  connected components = groups 0..U-1 (by smallest bucket); a random nonce per group
    5. SecAgg #3   pk of every bucket (sk_b = slot_key(T_s), as _pk_secagg); the Aggregator seals
                   (group, nonce) to every bucket's pk: only holders of a subset tag can open it
    6. clients     open one of their buckets -> group; sk = H(T_kw | nonce): only members know both
                   (the Aggregator has the nonce, not T_kw; another group with the same key lacks the nonce)
    7. SecAgg #4   pk of every group (_pk_secagg)
    Leakage beyond U: everyone learns M; the Aggregator also the bucket graph (how subset buckets
    co-occur in labels). Semi-honest, as the rest of the union.
    Returns (index per client {label: group}, sks per client {label: sk}, pks per group, stats)."""
    t0 = time.perf_counter()
    n = len(client_keys)
    units = []                                                          # per client {key: {subset}}
    for keys, sets in zip(client_keys, client_sets):
        u = {}
        for x, kw in keys.items():
            u.setdefault(kw, set()).update(subsets(sets[x], t))
        units.append(u)
    if any(not ss for u in units for ss in u.values()):
        raise ValueError(f'every label needs >= {t} image anchors')
    items = [list(u) + [kw + IMG + s for kw, ss in u.items() for s in sorted(ss)] for u in units]
    T = tags(items, None, workers)                                      # 1. one ring for everything
    tag_seconds = time.perf_counter() - t0
    sub = [{it: v for it, v in own.items() if IMG in it} for own in T]
    pos, M, stats = _union_from_tags(sub, workers, bucket_bits, session + b'/union')     # 2.
    sa = stats['secagg']
    tri = np.triu_indices(M, 1)
    E = len(tri[0])
    edge = lambda a, b: a * M - a * (a + 1) // 2 + (b - a - 1)
    vectors, first = {}, []
    for c, u in enumerate(units):                                       # 3. star edges per label
        v, f = np.zeros(E, np.uint64), {}
        for kw, ss in u.items():
            ps = sorted(pos[c][kw + IMG + s] for s in ss)
            f[kw] = ps[0]
            if len(ps) > 1:
                v[[edge(ps[0], p) for p in ps[1:]]] = np.frombuffer(secrets.token_bytes(8 * (len(ps) - 1)), np.uint64)
        vectors[c] = v
        first.append(f)
    total = (run_secagg(vectors, threshold=max(2, -(-2 * n // 3)), modulus_bits=64,
                        session=session + b'/edges', workers=workers, stats=sa)[0] if E else np.zeros(0, np.uint64))
    nz = np.flatnonzero(total)
    group = components(M, tri[0][nz], tri[1][nz])                       # 4. Aggregator
    U = max(group) + 1
    nonce = [secrets.token_bytes(16) for _ in range(U)]
    bucket_sk = [{pos[c][it]: slot_key(v) for it, v in own.items()} for c, own in enumerate(sub)]
    bucket_pk = _pk_secagg(bucket_sk, M, session + b'/bucket-pk', workers, sa)     # 5.
    sealed = [_seal(bucket_pk[b], b'pairs-group/%d' % b, struct.pack('>I', group[b]) + nonce[group[b]])
              for b in range(M)]                                        # posted on the BB
    index, sks = [], []
    for c, u in enumerate(units):                                       # 6. each client, locally
        gk = {}
        for kw in u:
            b = first[c][kw]
            msg = _open(bucket_sk[c][b], bucket_pk[b], b'pairs-group/%d' % b, sealed[b])
            g = struct.unpack('>I', msg[:4])[0]
            gk[kw] = (g, rg.hash_to_scalar(T[c][kw] + msg[4:], b'label-union-group-sk-v1'))
        index.append({x: gk[kw][0] for x, kw in client_keys[c].items()})
        sks.append({x: gk[kw][1] for x, kw in client_keys[c].items()})
    pks = _pk_secagg([{gk: sk for gk, sk in zip(idx.values(), sk_.values())} for idx, sk_ in zip(index, sks)],
                     U, session + b'/pk', workers, sa)                  # 7.
    m = max(len(i) for i in items)
    stats.update(seconds=time.perf_counter() - t0, tag_seconds=tag_seconds, image_t=t, buckets=M, edges=int(len(nz)),
                 ring_bytes_per_client=n * m * 32, edge_upload_bytes_per_client=8 * E,
                 sealed_download_bytes_per_client=sum(map(len, sealed)),
                 setup_upload_bytes_per_client=n * m * 32 + (sa['payload_up'] + sa['control_up']) / n,
                 setup_download_bytes_per_client=n * m * 32 + sa['control_down'] / n + M * 4
                 + sum(map(len, sealed)) + (M + U) * 32)
    return index, sks, pks, stats
