"""ristretto255 prime-order group (RFC 9496), pure Python, for OT and OPRF.

ponytail: pure Python and NOT constant time (double-and-add, Python ints).
Fine for a research prototype; use libsodium's crypto_core_ristretto255_* in deployment.
"""
import hashlib
import secrets

P = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493      # group order
D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
SQRT_M1 = 19681161376707505956807079304988542015446066515923890162744021073123829784752
SQRT_AD_MINUS_ONE = 25063068953384623474111414158702152701244531502492656460079210482610430750235
INVSQRT_A_MINUS_D = 54469307008909316920995813868745141605393597292927456921205312896311721017578
ONE_MINUS_D_SQ = (1 - D * D) % P
D_MINUS_ONE_SQ = (D - 1) ** 2 % P


def _neg(x):
    return x & 1                       # IS_NEGATIVE on the canonical representative


def _abs(x):
    return (-x) % P if _neg(x) else x


def _sqrt_ratio_m1(u, v):
    r = (u * pow(v, 3, P)) * pow(u * pow(v, 7, P), (P - 5) // 8, P) % P
    check = v * r * r % P
    correct = check == u % P
    flipped = check == (-u) % P
    flipped_i = check == (-u * SQRT_M1) % P
    if flipped or flipped_i:
        r = r * SQRT_M1 % P
    return correct or flipped, _abs(r)


class Point:
    """Extended twisted Edwards coordinates (X:Y:Z:T), a = -1."""
    __slots__ = ('X', 'Y', 'Z', 'T')

    def __init__(self, X, Y, Z, T):
        self.X, self.Y, self.Z, self.T = X % P, Y % P, Z % P, T % P

    def __add__(self, o):
        a = (self.Y - self.X) * (o.Y - o.X) % P
        b = (self.Y + self.X) * (o.Y + o.X) % P
        c = self.T * 2 * D * o.T % P
        d = self.Z * 2 * o.Z % P
        e, f, g, h = b - a, d - c, d + c, b + a
        return Point(e * f, g * h, f * g, e * h)

    def __neg__(self):
        return Point(-self.X, self.Y, self.Z, -self.T)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, k):
        k %= L
        result, addend = IDENTITY, self
        while k:
            if k & 1:
                result = result + addend
            addend = addend + addend
            k >>= 1
        return result

    __rmul__ = __mul__

    def __eq__(self, o):
        return (self.X * o.Y - self.Y * o.X) % P == 0 or (self.Y * o.Y - self.X * o.X) % P == 0

    def __hash__(self):
        return hash(self.encode())

    def encode(self) -> bytes:
        x0, y0, z0, t0 = self.X, self.Y, self.Z, self.T
        u1 = (z0 + y0) * (z0 - y0) % P
        u2 = x0 * y0 % P
        _, invsqrt = _sqrt_ratio_m1(1, u1 * u2 * u2 % P)
        den1, den2 = invsqrt * u1 % P, invsqrt * u2 % P
        z_inv = den1 * den2 * t0 % P
        if _neg(t0 * z_inv % P):
            x, y, den_inv = y0 * SQRT_M1 % P, x0 * SQRT_M1 % P, den1 * INVSQRT_A_MINUS_D % P
        else:
            x, y, den_inv = x0, y0, den2
        if _neg(x * z_inv % P):
            y = (-y) % P
        return _abs(den_inv * (z0 - y) % P).to_bytes(32, 'little')

    @staticmethod
    def decode(data: bytes) -> 'Point':
        if len(data) != 32:
            raise ValueError('ristretto255 encoding must be 32 bytes')
        s = int.from_bytes(data, 'little')
        if s >= P or _neg(s):
            raise ValueError('non-canonical ristretto255 encoding')
        ss = s * s % P
        u1, u2 = (1 - ss) % P, (1 + ss) % P
        u2_sqr = u2 * u2 % P
        v = (-(D * u1 * u1) - u2_sqr) % P
        was_square, invsqrt = _sqrt_ratio_m1(1, v * u2_sqr % P)
        den_x = invsqrt * u2 % P
        den_y = invsqrt * den_x * v % P
        x = _abs(2 * s * den_x % P)
        y = u1 * den_y % P
        t = x * y % P
        if not was_square or _neg(t) or y == 0:
            raise ValueError('invalid ristretto255 encoding')
        return Point(x, y, 1, t)


IDENTITY = Point(0, 1, 1, 0)


def _map(b: bytes) -> Point:
    t = int.from_bytes(b, 'little') & ((1 << 255) - 1)
    t %= P
    r = SQRT_M1 * t * t % P
    u = (r + 1) * ONE_MINUS_D_SQ % P
    v = (-1 - r * D) * (r + D) % P
    was_square, s = _sqrt_ratio_m1(u, v)
    s_prime = (-_abs(s * t % P)) % P
    s, c = (s, P - 1) if was_square else (s_prime, r)
    n = (c * (r - 1) * D_MINUS_ONE_SQ - v) % P
    w0, w1 = 2 * s * v % P, n * SQRT_AD_MINUS_ONE % P
    w2, w3 = (1 - s * s) % P, (1 + s * s) % P
    return Point(w0 * w3, w2 * w1, w1 * w3, w0 * w2)


def from_uniform(b: bytes) -> Point:
    """RFC 9496 element derivation from 64 uniform bytes."""
    if len(b) != 64:
        raise ValueError('need 64 uniform bytes')
    return _map(b[:32]) + _map(b[32:])


def expand_message_xmd(msg: bytes, dst: bytes, length: int) -> bytes:
    """RFC 9380 expand_message_xmd with SHA-512."""
    if len(dst) > 255 or length > 255 * 64:
        raise ValueError('dst or length too long')
    dst_prime = dst + bytes([len(dst)])
    b0 = hashlib.sha512(b'\0' * 128 + msg + length.to_bytes(2, 'big') + b'\0' + dst_prime).digest()
    out, bi = b'', b''
    for i in range(1, -(-length // 64) + 1):
        bi = hashlib.sha512((bytes(x ^ y for x, y in zip(b0, bi)) if bi else b0)
                            + bytes([i]) + dst_prime).digest()
        out += bi
    return out[:length]


def hash_to_group(msg: bytes, dst: bytes) -> Point:
    return from_uniform(expand_message_xmd(msg, dst, 64))


def hash_to_scalar(msg: bytes, dst: bytes) -> int:
    return int.from_bytes(expand_message_xmd(msg, dst, 64), 'little') % L


def random_scalar() -> int:
    return 1 + secrets.randbelow(L - 1)


BASE = Point.decode(bytes.fromhex('e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76'))
