"""Label union over the public dictionary by all clients in MPC (align.mpc, GMW). No Aggregator
in the computation; nobody learns who holds what.

Outputs
  Aggregator:  U only: its label table is the index list 0..U-1
  everyone:    U (the layout size)
  client i:    for each of its own labels, the index in [0, U); equal labels, equal index
Nobody learns which dictionary entries exist: indices are dense ranks in a secret random
order, so an index says only that it is < U.

Circuit (rows = dictionary positions, padded to a power of two)
  exists[d]  = OR_i ind_i[d]                       (n-1 AND layers over the client axis)
  sort 1     rows [tag | pos | exists] by a secret random tag (XOR of every client's input)
  rank       exclusive prefix sum of exists  ->  dense index in the shuffled order;  U = total
  sort 2     back to dictionary order by pos
  output     client i receives rank & ind_i (only its own positions are nonzero)

Cost: two bitonic sorts over the dictionary (~300k AND gates for 128 rows) plus n(n-1) base-OT
setups; independent of how many labels each client holds (PSI on a public universe).
ponytail: 30 clients ~ 20-30 CPU-min in pure Python, once, before training.
"""
import math
import secrets
import time

import numpy as np

from align.circuits import bitonic_sort, prefix_sum, to_bits, from_bits
from align.mpc import MPC, Shared, concat
from .discover import indicator

TAG_BITS = 40                      # random sort keys: collision prob ~ rows^2 / 2^41


def _or_reduce_rows(e, x: Shared) -> Shared:
    """OR over axis 0 (clients) with a log-depth tree."""
    while x.shape[0] > 1:
        if x.shape[0] % 2:
            x = concat([x, e.constant(np.zeros((1,) + x.shape[1:], np.uint8))], axis=0)
        h = x.shape[0] // 2
        x = ~e.and_(~x[:h], ~x[h:])
    return x[0]


def mpc_union(client_labels, dictionary, session=b'label-union', workers=1, shuffle=True):
    """Returns (index per client {label: index}, U, stats)."""
    n, D = len(client_labels), len(dictionary)
    if len(set(dictionary)) != D:
        raise ValueError('dictionary ids must be unique')
    rows = 1 << max(1, math.ceil(math.log2(D)))
    pbits, w = rows.bit_length(), (rows + 1).bit_length()
    t = time.perf_counter()
    e = MPC(n, session).warm(workers)
    setup_seconds = time.perf_counter() - t
    ind = [np.pad(indicator(l, dictionary).astype(np.uint8), (0, rows - D)) for l in client_labels]
    I = concat([e.input(v.reshape(1, rows), i) for i, v in enumerate(ind)], axis=0)       # (n, rows)
    exists = _or_reduce_rows(e, I).reshape(rows, 1)
    pos = e.constant(to_bits(np.arange(rows), pbits))                                      # MSB first
    if shuffle:
        tag = e.input(np.unpackbits(np.frombuffer(secrets.token_bytes(TAG_BITS * rows // 8 + 8), np.uint8))
                      [:rows * TAG_BITS].reshape(rows, TAG_BITS), 0)
        for i in range(1, n):
            tag = tag ^ e.input(np.unpackbits(np.frombuffer(secrets.token_bytes(TAG_BITS * rows // 8 + 8), np.uint8))
                                [:rows * TAG_BITS].reshape(rows, TAG_BITS), i)
        rec = bitonic_sort(e, concat([tag, pos, exists]), TAG_BITS)
        pos, exists = rec[:, TAG_BITS:TAG_BITS + pbits], rec[:, TAG_BITS + pbits:]
    # exclusive prefix sum: counts[k] = #existing rows before k in the (shuffled) order
    words = concat([e.constant(np.zeros((rows + 1, 1), np.uint8)), e.constant(np.zeros((rows + 1, w - 1), np.uint8))])
    words = words.set((slice(1, None), slice(0, 1)), exists)
    counts = prefix_sum(e, words)
    U = int(from_bits(e.open(counts[rows]), msb_first=False))
    rank = counts[:rows]                                                                   # LSB first
    if shuffle:
        rank = bitonic_sort(e, concat([pos, rank]), pbits)[:, pbits:]
    # mask with each client's SHARED indicator (private to it), all clients in one AND layer
    mask = Shared([np.broadcast_to(p[:, :, None], (n, rows, w)) for p in I.parts])
    masked = e.and_(mask, rank.reshape(1, rows, w).broadcast_to((n, rows, w)))
    at = {x: k for k, x in enumerate(dictionary)}
    out = []
    for i, labels in enumerate(client_labels):
        mine = e.reveal_to(masked[i], i)                                  # zero outside own rows
        out.append({x: int(from_bits(mine[at[x]], msb_first=False)) for x in labels})
    return out, U, dict(and_gates=e.and_gates, rounds=e.rounds, parties=n, rows=rows,
                        base_ot_seconds=setup_seconds, seconds=time.perf_counter() - t)
