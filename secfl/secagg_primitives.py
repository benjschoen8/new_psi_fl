"""SecAgg building blocks: X25519 -> HKDF pairwise seeds, ChaCha20 PRG, Shamir sharing."""
import secrets

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

SEED_BYTES = 32


def pairwise_seed(shared_secret: bytes, id_a, id_b, session: bytes) -> bytes:
    """Same 32-byte seed on both ends: HKDF-SHA256 over the X25519 output, bound to the pair."""
    lo, hi = sorted((str(id_a), str(id_b)))
    info = b'secagg/pair/' + lo.encode() + b'|' + hi.encode()
    return HKDF(hashes.SHA256(), SEED_BYTES, salt=session, info=info).derive(shared_secret)


def word_dtype(modulus_bits: int):
    """Smallest unsigned dtype holding Z_{2^modulus_bits}; its wrap-around is arithmetic mod 2^width."""
    return np.uint16 if modulus_bits <= 16 else np.uint32 if modulus_bits <= 32 else np.uint64


def prg(seed: bytes, length: int, modulus_bits: int = 32) -> np.ndarray:
    """Expand a seed into `length` uniform elements of Z_{2^modulus_bits}, in word_dtype(modulus_bits):
    2 / 4 / 8 ChaCha20 keystream bytes per element (16-bit sums need 4x less keystream than 64-bit)."""
    if len(seed) != SEED_BYTES or length < 0 or not 1 <= modulus_bits <= 64:
        raise ValueError('seed must be 32 bytes, length >= 0, 1 <= modulus_bits <= 64')
    dt = np.dtype(word_dtype(modulus_bits))
    enc = Cipher(algorithms.ChaCha20(seed, b'\0' * 16), mode=None).encryptor()
    words = np.frombuffer(enc.update(bytes(dt.itemsize * length)), dtype=dt.newbyteorder('<')).astype(dt)
    return words & dt.type((1 << modulus_bits) - 1) if modulus_bits < 8 * dt.itemsize else words


# ------------------------------------------------------------------ Shamir over GF(2^521 - 1)
PRIME = 2 ** 521 - 1          # Mersenne prime > 2^256: a 32-byte secret is one field element


def shamir_share(secret: bytes, threshold: int, xs):
    """Shares (x, y) of a <=64-byte secret; any `threshold` of them reconstruct it."""
    xs = list(xs)
    if not 1 <= threshold <= len(xs) or len(set(xs)) != len(xs) or any(not 0 < x < PRIME for x in xs):
        raise ValueError('need 1 <= threshold <= #shares and distinct nonzero x')
    if len(secret) > 64:
        raise ValueError('secret too long for one field element')
    coeffs = [int.from_bytes(secret, 'big')] + [secrets.randbelow(PRIME) for _ in range(threshold - 1)]
    def f(x):
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % PRIME
        return y
    return [(x, f(x)) for x in xs]


def shamir_reconstruct(shares, length: int = SEED_BYTES) -> bytes:
    """Lagrange interpolation at 0. Caller must pass at least `threshold` distinct shares."""
    if len({x for x, _ in shares}) != len(shares) or not shares:
        raise ValueError('need distinct shares')
    total = 0
    for i, (xi, yi) in enumerate(shares):
        num = den = 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                num = num * (-xj) % PRIME
                den = den * (xi - xj) % PRIME
        total = (total + yi * num * pow(den, -1, PRIME)) % PRIME
    if total >= 1 << (8 * length):
        raise ValueError('reconstruction out of range (too few or corrupted shares)')
    return total.to_bytes(length, 'big')
