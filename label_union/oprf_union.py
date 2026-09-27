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
import secrets
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
    chunks = [{x: [int(c) for c in np.frombuffer((rg.BASE * sk).encode(), '>u2')] for x, sk in own.items()}
              for own in keys]
    n = len(keys)
    for attempt in range(4):
        vectors = {}
        for i, idx in enumerate(index):
            v = np.zeros((U, PK_CHUNKS + 1), np.uint64)
            for x, cs in chunks[i].items():
                r = secrets.randbits(64)
                v[idx[x]] = [r] + [(r * c) % (1 << 64) for c in cs]
            vectors[i] = v.ravel()
        total, _ = run_secagg(vectors, threshold=max(2, -(-2 * n // 3)), modulus_bits=64,
                              session=session + b'/pk/%d' % attempt, workers=workers, stats=stats['secagg'])
        pks = [decode_pk([int(w) for w in row]) for row in total.reshape(U, PK_CHUNKS + 1)]
        if all(p is not None for p in pks):
            break
    else:
        raise RuntimeError('pk aggregation failed 4 times: holders disagree on a pk')
    m = max(len(l) for l in client_labels)
    sa = stats['secagg']
    stats.update(seconds=time.perf_counter() - t, pk_upload_bytes_per_client=8 * U * (PK_CHUNKS + 1),
                 # every padded list of m points (32 B) makes n hops: n*m*32 bytes sent per client
                 ring_bytes_per_client=n * m * 32,
                 setup_upload_bytes_per_client=n * m * 32 + (sa['payload_up'] + sa['control_up']) / n,
                 setup_download_bytes_per_client=n * m * 32 + sa['control_down'] / n + U * (4 + 32))   # + index list, pks
    return index, keys, pks, stats
