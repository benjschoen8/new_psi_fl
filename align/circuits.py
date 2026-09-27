"""A4b: circuit gadgets over align.mpc (n-party GMW). Bits live on the last axis.

Comparisons use MSB-first bits; arithmetic (add, prefix_sum) uses LSB-first bits.
Each gadget batches a whole layer into one AND call, so rounds = circuit depth.
"""
import numpy as np

from .mpc import Shared, concat


def and_reduce(e, x: Shared) -> Shared:
    """AND over the last axis, log depth."""
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = concat([x, e.constant(np.ones(x.shape[:-1] + (1,), np.uint8))])
        h = x.shape[-1] // 2
        x = e.and_(x[..., :h], x[..., h:])
    return x[..., 0]


def eq(e, a: Shared, b: Shared) -> Shared:
    return and_reduce(e, ~(a ^ b))


def lt(e, a: Shared, b: Shared) -> Shared:
    """[a < b] for unsigned MSB-first words; ~2K AND gates, log2(K)+1 rounds."""
    less, same = e.and_(~a, b), ~(a ^ b)
    k = less.shape[-1]
    width = 1 << max(0, (k - 1).bit_length())
    if width != k:                                   # pad on the LSB side: equal, not less
        pad = width - k
        less = concat([less, e.constant(np.zeros(less.shape[:-1] + (pad,), np.uint8))])
        same = concat([same, e.constant(np.ones(same.shape[:-1] + (pad,), np.uint8))])
    while less.shape[-1] > 1:
        h = less.shape[-1] // 2
        prod = e.and_(concat([same[..., 0::2], same[..., 0::2]]), concat([less[..., 1::2], same[..., 1::2]]))
        less, same = less[..., 0::2] ^ prod[..., :h], prod[..., h:]
    return less[..., 0]


def cond_swap(e, bit: Shared, x: Shared, y: Shared):
    """(y, x) where bit else (x, y), rows of width W; W AND gates per row."""
    diff = e.and_(bit.reshape(*bit.shape, 1).broadcast_to(x.shape), x ^ y)
    return x ^ diff, y ^ diff


def bitonic_sort(e, rec: Shared, key_bits: int) -> Shared:
    """Sort rows (N x W, N a power of 2) ascending by the first key_bits columns (MSB first)."""
    n = rec.shape[0]
    if n & (n - 1):
        raise ValueError('bitonic sort needs a power-of-two row count')
    k = 2
    while k <= n:
        j = k // 2
        while j:
            i = np.array([x for x in range(n) if (x ^ j) > x])
            l = i ^ j
            asc = (i & k) == 0
            first, second = np.where(asc, l, i), np.where(asc, i, l)
            swap = lt(e, rec[first, :key_bits], rec[second, :key_bits])
            new_i, new_l = cond_swap(e, swap, rec[i], rec[l])
            rec = rec.set(i, new_i).set(l, new_l)
            j //= 2
        k *= 2
    return rec


def add(e, a: Shared, b: Shared) -> Shared:
    """a + b mod 2^w, LSB-first words; w AND gates, w rounds (ripple carry)."""
    carry = e.constant(np.zeros(a.shape[:-1], np.uint8))
    out = []
    for t in range(a.shape[-1]):
        at, bt = a[..., t], b[..., t]
        out.append(at ^ bt ^ carry)
        carry = carry ^ e.and_(at ^ carry, bt ^ carry)
    return concat([o.reshape(*o.shape, 1) for o in out])


def prefix_sum(e, x: Shared) -> Shared:
    """Inclusive scan over axis 0 of LSB-first words (N x w); Hillis-Steele, log2(N) adds."""
    d = 1
    while d < x.shape[0]:
        x = x.set(slice(d, None), add(e, x[d:], x[:-d]))
        d *= 2
    return x


def to_bits(values, width, msb_first=True):
    v = np.asarray(values, np.uint64)[..., None] >> np.arange(width, dtype=np.uint64)
    bits = (v & np.uint64(1)).astype(np.uint8)
    return bits[..., ::-1] if msb_first else bits


def from_bits(bits, msb_first=True):
    bits = np.asarray(bits, np.uint64)
    if msb_first:
        bits = bits[..., ::-1]
    return (bits << np.arange(bits.shape[-1], dtype=np.uint64)).sum(-1)
