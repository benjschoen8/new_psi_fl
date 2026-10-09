"""SimHash form of the CSLS keyword test (pair version 'simhash' of mpspdz_pairwise).

Each client maps its fixed-point embedding e to k sign bits c = [P e >= 0] for k public random
hyperplanes P (public seed). For unit vectors the angle is estimated by theta ~ pi Ham(c, c') / k,
so CSLS 2 cos(theta) - r - r' >= tau becomes theta <= arccos((tau + r + r') / 2). The arccos is
linearised at a public u0 (default 1.0, i.e. r ~ 0.45), which makes the threshold additive:

    S Ham(c, c') + g + g' <= C,  g = round(S k c1 r / pi),  C = round(S k (theta0 - c1 (tau - u0)) / pi)

with theta0 = arccos(u0 / 2), c1 = 1 / (2 sqrt(1 - u0^2 / 4)). The garbled circuit evaluates only
XOR, a popcount and one comparison per pair of rows. This is an approximation of CSLS: compare
groupings with setup_smoke_hybrid --approx-check (plaintext, no MPC).
"""
import functools
import math

import numpy as np

BITS = 256                     # hyperplanes (multiple of 64)
SCALE = 16                     # S, power of two
U0 = 1.0
SEED = 20261009
G_BITS = 16


@functools.lru_cache(maxsize=8)
def planes(d, k=BITS, seed=SEED):
    return np.random.default_rng(seed).standard_normal((k, d))


def code(e, k=BITS):
    """0/1 vector of k sign bits of an (integer) embedding."""
    return (planes(len(e), k) @ np.asarray(e, float) >= 0).astype(np.int64)


def words(bits):
    """k bits -> k/64 integers, bit b of word w = bits[64 w + b]."""
    return [sum(int(b) << i for i, b in enumerate(bits[w:w + 64])) for w in range(0, len(bits), 64)]


def threshold(tau, k=BITS, S=SCALE, u0=U0):
    c1 = 1 / (2 * math.sqrt(1 - u0 * u0 / 4))
    return c1, int(round(S * k * (math.acos(u0 / 2) - c1 * (tau - u0)) / math.pi))


def g_share(r_fixed, k=BITS, S=SCALE, u0=U0, fix=7):
    """Integer threshold share of a row with CSLS hub term r (fixed point at 2^(2 fix))."""
    c1, _ = threshold(0, k, S, u0)
    return min(max(0, int(round(S * k * c1 * (r_fixed / (1 << 2 * fix)) / math.pi))), (1 << G_BITS) - 1)


def kw_match(a, b, tau, k=BITS, S=SCALE, u0=U0):
    """Plaintext predicate of the circuit (ideal functionality of pair version 'simhash')."""
    if a[0] == 'name' or b[0] == 'name':
        return a == b
    if a[0] == 'sym' or b[0] == 'sym':
        raise ValueError('simhash supports embeddings only')
    ham = int(np.sum(code(a[1], k) != code(b[1], k)))
    return S * ham + g_share(a[2], k, S, u0) + g_share(b[2], k, S, u0) <= threshold(tau, k, S, u0)[1]


def agreement(exact, approx):
    """Pair-level agreement of two groupings (group id per row) of the same rows."""
    N = len(exact)
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    together = lambda g, a, b: g[a] == g[b]
    return dict(rows=N, groups_exact=len(set(exact)), groups_approx=len(set(approx)),
                identical=list(exact) == list(approx),
                together_both=sum(together(exact, a, b) and together(approx, a, b) for a, b in pairs),
                merged_extra=sum(not together(exact, a, b) and together(approx, a, b) for a, b in pairs),
                split_extra=sum(together(exact, a, b) and not together(approx, a, b) for a, b in pairs))


def compare(rows, owners, tau, t=2, **kw):
    """Exact CSLS grouping vs SimHash grouping of the same rows."""
    from label_union.circuit_union import group
    exact, _ = group(rows, tau, t, owners)
    approx, _ = group(rows, tau, t, owners, kw_match=lambda a, b, tau: kw_match(a, b, tau, **kw))
    return agreement(exact, approx)
