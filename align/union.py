"""A4c: exact label union by all clients in MPC (align.mpc). Output: each client learns,
for each of its own labels, a slot index; equal names get equal slots, distinct names
distinct slots. Nobody learns anything else: not other clients' labels, not who shares
a label with whom, not the union itself.

Circuit (all n clients are the MPC parties):
  input   client i: m_max rows [invalid | H(name)] (padding rows: invalid=1, random hash);
          row position p = i*m_max + j is public and rides along as payload
  sort 1  bitonic sort by [invalid | hash]: valid rows first, equal names adjacent
  first   f_k = valid_k AND NOT (hash_k == hash_{k-1})
  rank    S_k = prefix sum of f (1-based index of the name among distinct names)
  sort 2  bitonic sort by position: every row returns to its owner's slot in the array
  output  S_p - 1 revealed to the owner of row p only

Names come from the public dictionary (exact PSI setting); H is a public hash, but the
hashes are only ever secret-shared. Hash width sigma = 40 + 2 log2 N (collision bound).
Slot numbering: dense ranks 0..U-1 by default (a client whose largest slot is s learns
U > s). hide_count=True maps ranks through a secret random permutation of [0, M), M = rows,
jointly sampled in the circuit, so a slot value says nothing about U.
"""
import hashlib
import math
import secrets
import time

import numpy as np

from .circuits import add, bitonic_sort, eq, lt, prefix_sum, to_bits, from_bits
from .mpc import MPC, concat

STAT_SECURITY = 40


def _hash_bits(name, sigma):
    digest = hashlib.sha256(b'label-union/' + str(name).encode('utf-8')).digest()
    return np.unpackbits(np.frombuffer(digest, np.uint8))[:sigma]


def _check(client_labels, m_max, engine, tag):
    n = len(client_labels)
    if n < 2 or any(len(set(l)) != len(l) or len(l) > m_max for l in client_labels):
        raise ValueError('need >= 2 clients, each with <= m_max distinct labels')
    engine = engine or MPC(n, tag)
    if engine.n != n:
        raise ValueError('engine must have one party per client')
    return n, engine


def _random_permutation(engine, n_rows, pbits):
    """Secret uniform permutation of [0, n_rows) as rows of LSB-first values (n_rows x pbits).

    Every party inputs a random 64-bit tag per index; tags are XORed (random if any party
    is honest) and the indices are sorted by tag inside the circuit.
    """
    tags = engine.input(np.unpackbits(np.frombuffer(secrets.token_bytes(8 * n_rows), np.uint8)).reshape(n_rows, 64), 0)
    for i in range(1, engine.n):
        tags = tags ^ engine.input(np.unpackbits(np.frombuffer(secrets.token_bytes(8 * n_rows), np.uint8)).reshape(n_rows, 64), i)
    rec = concat([tags, engine.constant(to_bits(np.arange(n_rows), pbits, msb_first=False))], axis=1)
    return bitonic_sort(engine, rec, 64)[:, 64:]


def _select(engine, index: 'Shared', table: 'Shared', ibits: int):
    """Oblivious table[index] per row: one-hot of index over table rows, then select."""
    n_idx, n_tab, width = index.shape[0], table.shape[0], table.shape[1]
    targets = engine.constant(to_bits(np.arange(n_tab), ibits, msb_first=False))
    onehot = eq(engine, index.reshape(n_idx, 1, ibits).broadcast_to((n_idx, n_tab, ibits)),
                targets.reshape(1, n_tab, ibits).broadcast_to((n_idx, n_tab, ibits)))
    return engine.and_(onehot.reshape(n_idx, n_tab, 1).broadcast_to((n_idx, n_tab, width)),
                       table.reshape(1, n_tab, width).broadcast_to((n_idx, n_tab, width))).xor_reduce(axis=1)


def _union_circuit(engine, client_labels, m_max, hide_count=False, max_labels=None):
    """Sort 1 -> first flags -> ranks -> sort 2 [-> secret permutation].

    Returns (slot of each row by position, LSB-first shared; meta).
    """
    n = len(client_labels)
    n_rows = 1 << math.ceil(math.log2(n * m_max))
    sigma = STAT_SECURITY + 2 * math.ceil(math.log2(n_rows))
    pbits = max(1, math.ceil(math.log2(n_rows)))
    wbits = math.ceil(math.log2(n_rows + 1))
    keys = []
    for i, labels in enumerate(client_labels):              # each client inputs its own rows
        rows = np.zeros((m_max, 1 + sigma), np.uint8)
        for j in range(m_max):
            if j < len(labels):
                rows[j, 1:] = _hash_bits(labels[j], sigma)
            else:
                rows[j, 0] = 1
                rows[j, 1:] = np.unpackbits(np.frombuffer(secrets.token_bytes(sigma // 8 + 1), np.uint8))[:sigma]
        keys.append(engine.input(rows, owner=i))
    filler = n_rows - n * m_max                            # public padding to a power of two
    if filler:
        pad = np.zeros((filler, 1 + sigma), np.uint8)
        pad[:, 0] = 1
        keys.append(engine.constant(pad))
    key = concat(keys, axis=0)
    rec = concat([key, engine.constant(to_bits(np.arange(n_rows), pbits))], axis=1)
    rec = bitonic_sort(engine, rec, 1 + sigma)                              # sort 1
    valid, h = ~rec[:, 0], rec[:, 1:1 + sigma]
    new = engine.and_(valid[1:], ~eq(engine, h[1:], h[:-1]))
    first = concat([valid[:1].reshape(1), new], axis=0)                     # f_k
    ranks = prefix_sum(engine, concat([first.reshape(n_rows, 1),
                                       engine.constant(np.zeros((n_rows, wbits - 1), np.uint8))], axis=1))
    rec2 = concat([rec[:, 1 + sigma:], ranks], axis=1)                     # [position | S]
    rec2 = bitonic_sort(engine, rec2, pbits)                               # sort 2: back to owners
    minus_one = engine.constant(np.ones((n_rows, wbits), np.uint8))       # two's complement -1
    slots = add(engine, rec2[:, pbits:], minus_one)[:, :pbits]             # dense slot = S - 1
    # public slot space M (power of two, <= rows); abort, revealing one bit, if U > M
    M = n_rows if max_labels is None else min(n_rows, 1 << math.ceil(math.log2(max_labels)))
    mbits = max(1, math.ceil(math.log2(M)))
    union_size = ranks[-1:]                                                # S of the last row = U
    fits = lt(engine, union_size[:, ::-1], engine.constant(to_bits(np.array([M + 1]), wbits)))
    if not engine.open(fits)[0]:
        raise ValueError(f'union larger than the public slot bound M={M}')
    if hide_count:
        slots = _select(engine, slots, _random_permutation(engine, M, mbits), pbits)
    else:
        slots = slots[:, :mbits] if mbits <= pbits else slots
    return slots, dict(rows=n_rows, slots=M, sigma=sigma, pbits=pbits, sbits=slots.shape[1], hide_count=hide_count)


def _own_rows(i, m_max, labels):
    return slice(i * m_max, i * m_max + len(labels))


def exact_union(client_labels, m_max: int, engine: MPC = None, hide_count=False, max_labels=None):
    """client_labels: list (one per client) of label names. Returns (slots, stats).

    slots[i] = {name: slot} is what client i learns. stats: gates, rounds, seconds.
    """
    n, engine = _check(client_labels, m_max, engine, b'exact-union')
    start, gates0, rounds0 = time.perf_counter(), engine.and_gates, engine.rounds
    shared_slots, meta = _union_circuit(engine, client_labels, m_max, hide_count, max_labels)
    slots = []
    for i, labels in enumerate(client_labels):
        s = from_bits(engine.reveal_to(shared_slots[_own_rows(i, m_max, labels)], party=i), msb_first=False)
        slots.append({name: int(v) for name, v in zip(labels, s)})
    stats = dict(and_gates=engine.and_gates - gates0, rounds=engine.rounds - rounds0,
                 seconds=time.perf_counter() - start, **meta)
    return slots, stats


def exact_union_with_keys(client_labels, m_max: int, engine: MPC = None, hide_count=False, max_labels=None):
    """A5: union + one ristretto255 key pair per slot, no party ever holding a whole sk.

    Slots live in the public space [0, M), M = max_labels rounded up to a power of two
    (default: the padded row count); the circuit aborts with one public bit if U > M.
      1. Every client i draws x_{i,s} for each slot s and publishes X_{i,s} = x_{i,s} B.
         pk_s = sum_i X_{i,s} is computable by anyone (published on the BB).
      2. In the circuit: sk_s = sum_i x_{i,s} (integer sum; sk_s B = pk_s without reduction),
         then each row looks up sk of ITS slot obliviously (one-hot over slots) and only
         the row's owner sees the result.
    Holders of slot s all get the same sk_s; nobody else learns it, not even a subset of
    fewer than n colluding clients.
    Returns (slots, keys, pks, stats): keys[i] = {name: sk int}, pks = [encoded pk_s].
    """
    from secfl import ristretto as rg
    n, engine = _check(client_labels, m_max, engine, b'exact-union-keys')
    start, gates0, rounds0 = time.perf_counter(), engine.and_gates, engine.rounds
    shared_slots, meta = _union_circuit(engine, client_labels, m_max, hide_count, max_labels)
    n_slots, sbits = meta['slots'], meta['sbits']
    xbits = rg.L.bit_length()
    width = xbits + math.ceil(math.log2(n)) + 1

    # step 1: contributions and public keys
    contrib = [[rg.random_scalar() for _ in range(n_slots)] for _ in range(n)]
    pks = []
    for s in range(n_slots):
        pk = rg.IDENTITY
        for i in range(n):
            pk = pk + rg.BASE * contrib[i][s]
        pks.append(pk.encode())

    # step 2a: sk_s = sum_i x_{i,s} inside the circuit (LSB-first integers)
    def as_bits(values):
        return np.stack([np.array([(v >> t) & 1 for t in range(width)], np.uint8) for v in values])
    total = engine.input(as_bits(contrib[0]), owner=0)
    for i in range(1, n):
        total = add(engine, total, engine.input(as_bits(contrib[i]), owner=i))     # n_slots x width

    # step 2b: every row obliviously picks sk of its own slot
    picked = _select(engine, shared_slots, total, sbits)

    slots, keys = [], []
    for i, labels in enumerate(client_labels):
        rows = _own_rows(i, m_max, labels)
        s = from_bits(engine.reveal_to(shared_slots[rows], party=i), msb_first=False)
        sk_bits = engine.reveal_to(picked[rows], party=i)
        slots.append({name: int(v) for name, v in zip(labels, s)})
        keys.append({name: sum(int(b) << t for t, b in enumerate(row)) % rg.L
                     for name, row in zip(labels, sk_bits)})
    stats = dict(and_gates=engine.and_gates - gates0, rounds=engine.rounds - rounds0,
                 seconds=time.perf_counter() - start, **meta)
    return slots, keys, pks, stats
