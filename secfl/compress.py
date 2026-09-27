"""Smaller SecAgg uploads for the per-label pipeline (secure_main), privacy unchanged: every client
still sends one dense vector over ALL slots (zeros for labels it does not hold).

  keep      public per-round random subset of coordinates (same for every client and slot, from
            a public seed): only those coordinates of each slot's update are uploaded; the rest keep
            the global value this round. Vector length x keep_frac.
  quantize  each kept coordinate -> integer in [-127, 127] (8 bits), stochastic rounding (unbiased)
            with a public scale c per slot AND per tensor (one scale for the whole generator lets
            BatchNorm statistics dominate: ~90% error on real DCGAN updates; per tensor: <1%). With n clients the sum lies in [-127 n, 127 n], so a
            16-bit SecAgg modulus holds it for n <= 258: 8 bytes -> 2 bytes per coordinate.
  counts    one extra word per slot: 1 if the client holds it. The Aggregator divides by the count
            (plain mean of the holders' updates), as secure_main already reveals holder counts.

Upload per client: 2 bytes x (M x keep x |G| + M). Security: SecAgg masks are uniform mod 2^16,
so a smaller modulus hides the values exactly as well; nothing about who holds what changes.
"""
import hashlib

import numpy as np

BITS = 16
LEVELS = 127


def keep_index(size, frac, round_index, session=b''):
    """Sorted public coordinate subset of a slot vector for this round."""
    if frac >= 1:
        return np.arange(size)
    seed = int.from_bytes(hashlib.sha256(b'keep/%d/' % round_index + session).digest()[:8], 'big')
    k = max(1, int(round(size * frac)))
    return np.sort(np.random.default_rng(seed).choice(size, k, replace=False))


def quantize(u, c, rng):
    """u / c * 127 -> stochastic rounding -> int in [-127, 127]."""
    x = np.clip(np.asarray(u, np.float64) / c, -1, 1) * LEVELS
    lo = np.floor(x)
    return (lo + (rng.random(x.shape) < (x - lo))).astype(np.int64)


def encode(updates, M, k, scales, rng):
    """updates {slot: kept coordinates of the update} -> uint64 vector (mod 2^16) of length M*k + M."""
    vec = np.zeros(M * k + M, np.int64)
    for s, u in updates.items():
        vec[s * k:(s + 1) * k] = quantize(u, scales[s], rng)
        vec[M * k + s] = 1
    return (vec % (1 << BITS)).astype(np.uint64)


def decode(total, M, k, scales):
    """Sum vector -> ({slot: mean update over holders, kept coordinates}, {slot: holders})."""
    v = np.asarray(total, np.int64) % (1 << BITS)
    v = np.where(v >= 1 << (BITS - 1), v - (1 << BITS), v)                 # signed
    counts = v[M * k:]
    means = {s: v[s * k:(s + 1) * k] / LEVELS * scales[s] / counts[s] for s in range(M) if counts[s] > 0}
    return means, {s: int(counts[s]) for s in range(M)}


def _blocks(spec):
    off = 0
    for _, shape, _ in spec:
        n = int(np.prod(shape))
        yield slice(off, off + n)
        off += n


def initial_scales(spec, flat_init, c0=0.05):
    """Per-coordinate scale for round 1 (public): max(c0, RMS of the tensor's public init / 2)."""
    out = np.empty(flat_init.size)
    for b in _blocks(spec):
        out[b] = max(c0, .5 * float(np.sqrt(np.mean(flat_init[b] ** 2))))
    return out


def next_scales(spec, delta, kept, floor=1e-4, factor=4.0):
    """Per-coordinate scale for the next round: factor x RMS of each tensor's last global change,
    measured on the coordinates that were uploaded (holders and the Aggregator both know it).
    Clipping keeps rare outliers bounded."""
    mask = np.zeros(delta.size, bool)
    mask[kept] = True
    out = np.empty(delta.size)
    for b in _blocks(spec):
        d = delta[b][mask[b]]
        out[b] = max(floor, factor * float(np.sqrt(np.mean(d * d)))) if d.size else floor
    return out
