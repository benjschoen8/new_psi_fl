"""A3: two-party exact circuit-PSI. Nobody learns the intersection; membership stays XOR-shared.

Receiver R (party 0) holds X (n items), sender S (party 1) holds Y (m items) and an OPRF key.
  1. OPRF (secfl.oprf): R obtains F_k(x) blindly; S computes F_k(y). R learns only PRF
     values of its own items, which say nothing about Y.
  2. 2PC (align.twopc): R inputs trunc_sigma(F_k(x_a)), S inputs trunc_sigma(F_k(y_b));
     eq[a, b] = [F(x_a) == F(y_b)]; member[a] = XOR_b eq[a, b] (at most one b matches).
Output: shares of member[a] (alignment version, "secret-shared table" row of R).

Key version: S also inputs a payload P_b (e.g. K of that label) and fresh random R_a.
  out[a] = member[a] ? XOR_b eq[a,b]&P_b : R_a, revealed to R only.
R gets P_b for its intersecting items and an unrelated random string otherwise, so it
cannot tell which is which from the value alone, and it has no choice bit to cheat with.

Cost: n*m*(sigma-1) AND gates (+ n*m*len(P) + n*len(P) for keys); sigma = 40 + log2(n*m).
ponytail: all-pairs comparison, fine for label sets (tens of items); cuckoo hashing +
OPPRF brings it to linear for large sets.
"""
import math
import secrets

import numpy as np

from secfl import oprf, ristretto as rg
from .psi import encode
from .twopc import TwoPC

STAT_SECURITY = 40


def _bits(data: bytes, nbits: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, np.uint8))[:nbits]


def _oprf_values(receiver_items, sender_items, sk):
    """Step 1: R's F_k(x) through blind evaluation; S's F_k(y) locally."""
    fx = []
    for x in receiver_items:
        r, q = oprf.blind(encode(x))
        fx.append(oprf.finalize(encode(x), r, oprf.blind_evaluate(sk, q)))
    fy = [oprf.evaluate(sk, encode(y)) for y in sender_items]
    return fx, fy


def circuit_psi(receiver_items, sender_items, sender_payloads=None, pc: TwoPC = None):
    """Returns dict(member=Shared (n,), [keys=list of bytes revealed to R], pc=TwoPC, sigma).

    sender_payloads: optional list aligned with sender_items, equal-length bytes (key version).
    """
    X, Y = list(dict.fromkeys(receiver_items)), list(dict.fromkeys(sender_items))
    if len(X) != len(receiver_items) or len(Y) != len(sender_items) or not X or not Y:
        raise ValueError('both sets must be nonempty and duplicate-free')
    n, m = len(X), len(Y)
    sigma = STAT_SECURITY + math.ceil(math.log2(n * m + 1))
    pc = pc or TwoPC(b'circuit-psi')
    fx, fy = _oprf_values(X, Y, rg.random_scalar())
    ex = pc.input(np.stack([_bits(v, sigma) for v in fx]), owner=0)[:, None, :]      # n x 1 x sigma
    ey = pc.input(np.stack([_bits(v, sigma) for v in fy]), owner=1)[None, :, :]      # 1 x m x sigma
    eq = pc.eq(ex.broadcast_to((n, m, sigma)), ey.broadcast_to((n, m, sigma)))      # n x m
    out = dict(member=eq.xor_reduce(axis=1), pc=pc, sigma=sigma)
    if sender_payloads is not None:
        if len(sender_payloads) != m or len({len(p) for p in sender_payloads}) != 1:
            raise ValueError('one equal-length payload per sender item')
        nbits = 8 * len(sender_payloads[0])
        P = pc.input(np.stack([_bits(p, nbits) for p in sender_payloads]), owner=1)  # m x nbits
        picked = pc.and_(eq.reshape(n, m, 1).broadcast_to((n, m, nbits)),
                         P[None].broadcast_to((n, m, nbits))).xor_reduce(axis=1)    # n x nbits
        noise = pc.input(np.stack([_bits(secrets.token_bytes(nbits // 8), nbits) for _ in range(n)]), owner=1)
        bits = pc.reveal_to(pc.mux(out['member'], picked, noise), party=0)
        out['keys'] = [np.packbits(row).tobytes() for row in bits]
    return out
