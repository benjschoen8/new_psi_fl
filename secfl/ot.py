"""1-out-of-2 oblivious transfer: Simplest OT (base) + IKNP extension. Semi-honest.

Message flow is explicit (msg1, msg2, ...) so parties can later run on separate
machines; nothing here does networking. Messages m0, m1 are equal-length bytes.

Simplest OT: Chou & Orlandi, LATINCRYPT 2015, over ristretto255.
IKNP: Ishai, Kilian, Nissim, Petrank, CRYPTO 2003, kappa = 128.
"""
import hashlib
import secrets

import numpy as np

from . import ristretto as rg

KAPPA = 128


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _kdf(tag: bytes, *parts: bytes, n: int) -> bytes:
    h = hashlib.shake_256(tag)
    for p in parts:
        h.update(len(p).to_bytes(4, 'big') + p)
    return h.digest(n)


def _check_pairs(pairs):
    if not pairs or any(len(m0) != len(m1) for m0, m1 in pairs):
        raise ValueError('need nonempty pairs of equal-length messages')


# ------------------------------------------------------------------ Simplest OT
class BaseOTSender:
    """Holds pairs (m0_j, m1_j). One key A for the whole batch; index j separates OTs."""

    def __init__(self, session: bytes = b''):
        self.session = session
        self._a = rg.random_scalar()
        self.A = rg.BASE * self._a

    def msg1(self) -> bytes:
        return self.A.encode()

    def msg3(self, receiver_points, pairs):
        _check_pairs(pairs)
        if len(receiver_points) != len(pairs):
            raise ValueError('one receiver point per OT')
        out, a_enc = [], self.A.encode()
        for j, (enc, (m0, m1)) in enumerate(zip(receiver_points, pairs)):
            R = rg.Point.decode(enc)
            idx = j.to_bytes(8, 'big')
            k0 = _kdf(b'sot', self.session, idx, a_enc, enc, (R * self._a).encode(), n=len(m0))
            k1 = _kdf(b'sot', self.session, idx, a_enc, enc, ((R - self.A) * self._a).encode(), n=len(m1))
            out.append((_xor(m0, k0), _xor(m1, k1)))
        return out


class BaseOTReceiver:
    def __init__(self, choices, session: bytes = b''):
        if any(c not in (0, 1) for c in choices):
            raise ValueError('choices must be bits')
        self.choices, self.session = list(choices), session
        self._b = [rg.random_scalar() for _ in self.choices]

    def msg2(self, A_enc: bytes):
        self._A_enc = A_enc
        A = rg.Point.decode(A_enc)
        if A == rg.IDENTITY:
            raise ValueError('sender point must not be the identity')
        self._A = A
        return [(rg.BASE * b + (A if c else rg.IDENTITY)).encode()
                for b, c in zip(self._b, self.choices)]

    def output(self, ciphertexts, receiver_points):
        out = []
        for j, (b, c, (e0, e1), enc) in enumerate(zip(self._b, self.choices, ciphertexts, receiver_points)):
            e = e1 if c else e0
            k = _kdf(b'sot', self.session, j.to_bytes(8, 'big'), self._A_enc, enc,
                     (self._A * b).encode(), n=len(e))
            out.append(_xor(e, k))
        return out


# ------------------------------------------------------------------ IKNP extension
def _prg_bits(seed: bytes, m: int, session: bytes) -> np.ndarray:
    raw = _kdf(b'iknp-prg', session, seed, n=(m + 7) // 8)
    return np.unpackbits(np.frombuffer(raw, np.uint8))[:m]


def _row_bytes(bits_row: np.ndarray) -> bytes:
    return np.packbits(bits_row).tobytes()


class IKNPReceiver:
    """Extension receiver with choice bits r. Acts as base-OT *sender* of seed pairs."""

    def __init__(self, choices, session: bytes = b''):
        self.r = np.array(choices, dtype=np.uint8)
        if self.r.ndim != 1 or not len(self.r) or not np.isin(self.r, (0, 1)).all():
            raise ValueError('choices must be a nonempty bit list')
        self.session = session
        self._seeds = [(secrets.token_bytes(16), secrets.token_bytes(16)) for _ in range(KAPPA)]
        self._base = BaseOTSender(session + b'/base')

    def msg1(self) -> bytes:
        return self._base.msg1()

    def msg3(self, receiver_points):
        return self._base.msg3(receiver_points, self._seeds)

    def msg5(self) -> bytes:
        """u_i = G(k_i^0) xor G(k_i^1) xor r for each of the kappa columns."""
        m = len(self.r)
        self._t = np.stack([_prg_bits(k0, m, self.session) for k0, _ in self._seeds])   # kappa x m
        u = np.stack([self._t[i] ^ _prg_bits(k1, m, self.session) ^ self.r
                      for i, (_, k1) in enumerate(self._seeds)])
        return np.packbits(u, axis=1).tobytes()

    def output(self, ys):
        if len(ys) != len(self.r):
            raise ValueError('one ciphertext pair per OT')
        rows = self._t.T                                                                 # m x kappa
        return [_xor(ys[j][self.r[j]], _kdf(b'iknp-h', self.session, j.to_bytes(8, 'big'),
                                             _row_bytes(rows[j]), n=len(ys[j][0])))
                for j in range(len(self.r))]


class IKNPSender:
    """Extension sender with pairs (x0_j, x1_j). Acts as base-OT *receiver* with secret s."""

    def __init__(self, session: bytes = b''):
        self.session = session
        self.s = np.frombuffer(secrets.token_bytes(KAPPA // 8), np.uint8)
        self.s = np.unpackbits(self.s)
        self._base = BaseOTReceiver(self.s.tolist(), session + b'/base')

    def msg2(self, A_enc: bytes):
        self._points = self._base.msg2(A_enc)
        return self._points

    def msg4(self, base_ciphertexts):
        self._seeds = self._base.output(base_ciphertexts, self._points)

    def msg6(self, u_bytes: bytes, pairs):
        _check_pairs(pairs)
        m = len(pairs)
        u = np.unpackbits(np.frombuffer(u_bytes, np.uint8).reshape(KAPPA, -1), axis=1)[:, :m]
        q = np.stack([_prg_bits(k, m, self.session) ^ (u[i] if self.s[i] else 0)
                      for i, k in enumerate(self._seeds)]).astype(np.uint8)              # kappa x m
        rows = q.T                                                                       # q_j = t_j xor r_j s
        out = []
        for j, (x0, x1) in enumerate(pairs):
            idx = j.to_bytes(8, 'big')
            out.append((_xor(x0, _kdf(b'iknp-h', self.session, idx, _row_bytes(rows[j]), n=len(x0))),
                        _xor(x1, _kdf(b'iknp-h', self.session, idx, _row_bytes(rows[j] ^ self.s), n=len(x1)))))
        return out


def run_iknp(pairs, choices, session: bytes = b''):
    """Local driver of the 6-message flow; returns what the receiver learns."""
    receiver, sender = IKNPReceiver(choices, session), IKNPSender(session)
    points = sender.msg2(receiver.msg1())
    sender.msg4(receiver.msg3(points))
    return receiver.output(sender.msg6(receiver.msg5(), pairs))


def run_base_ot(pairs, choices, session: bytes = b''):
    sender, receiver = BaseOTSender(session), BaseOTReceiver(choices, session)
    points = receiver.msg2(sender.msg1())
    return receiver.output(sender.msg3(points, pairs), points)


# ------------------------------------------------------------------ vectorized bit OT (for 2PC)
def _pack_columns(cols: np.ndarray) -> np.ndarray:
    """kappa x m bit matrix -> m x 16 bytes: row j = the 128 bits of column j, MSB first.
    Packs 8 kappa-rows at a time (5x faster than transposing the bit matrix first)."""
    k, m = cols.shape
    b = cols.reshape(k // 8, 8, m)
    z = np.zeros((k // 8, m), np.uint8)
    for bit in range(8):
        z |= b[:, bit, :] << (7 - bit)
    return np.ascontiguousarray(z.T)


def _crh_bits(cols: np.ndarray, key: bytes) -> np.ndarray:
    """Correlation-robust hash of each 128-bit column (one per OT) to one bit: fixed-key AES MMO,
    tweaked by index. cols: kappa x m."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    m = cols.shape[1]
    blk = _pack_columns(cols)                                                # m x 16 bytes
    tweak = np.zeros((m, 16), np.uint8)
    tweak[:, 8:] = np.arange(m, dtype='>u8').view(np.uint8).reshape(m, 8)
    blk = blk ^ tweak
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(blk.tobytes())
    return (np.frombuffer(enc, np.uint8).reshape(m, 16)[:, 0] ^ blk[:, 0]) & 1


def _crh_words(cols: np.ndarray, key: bytes) -> np.ndarray:
    """As _crh_bits, but 64 output bits per OT."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    m = cols.shape[1]
    blk = _pack_columns(cols)
    tweak = np.zeros((m, 16), np.uint8)
    tweak[:, 8:] = np.arange(m, dtype='>u8').view(np.uint8).reshape(m, 8)
    blk = blk ^ tweak
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(blk.tobytes())
    return (np.frombuffer(enc, np.uint8).reshape(m, 16)[:, :8] ^ blk[:, :8]).copy().view(np.uint64)[:, 0]


class IKNPBitSession:
    """Bit-OT extension with base OTs done once; each transfer() uses fresh PRG streams.

    Roles are fixed per session (who sends, who chooses). The kappa base OTs cost
    ~0.1 s in pure Python, so a 2PC keeps one session per direction and extends from it.
    """

    def __init__(self, session: bytes = b''):
        self.session, self.counter = session, 0
        receiver, sender = IKNPReceiver([0], session), IKNPSender(session)
        points = sender.msg2(receiver.msg1())
        sender.msg4(receiver.msg3(points))
        self._rseeds, self._s, self._sseeds = receiver._seeds, sender.s, sender._seeds
        self._key = hashlib.sha256(b'iknp-crh' + session).digest()[:16]      # public fixed key

    def _extend(self, c):
        """IKNP extension for choice bits c -> (sender q, receiver t, hash key), both kappa x m:
        column j of q = column j of t ^ c_j s."""
        m, tag = len(c), self.session + b'/ext/' + self.counter.to_bytes(8, 'big')
        self.counter += 1
        # receiver: t_i = G(k_i^0), u_i = t_i ^ G(k_i^1) ^ c          (kappa x m)
        t = np.stack([_prg_bits(k0, m, tag) for k0, _ in self._rseeds])
        u = np.stack([t[i] ^ _prg_bits(k1, m, tag) ^ c for i, (_, k1) in enumerate(self._rseeds)])
        # sender: q_i = G(k_i^{s_i}) ^ s_i u_i  =>  row q_j = t_j ^ c_j s
        q = np.stack([_prg_bits(k, m, tag) ^ (u[i] if self._s[i] else 0)
                      for i, k in enumerate(self._sseeds)]).astype(np.uint8)
        return q, t.astype(np.uint8), hashlib.sha256(self._key + tag).digest()[:16]

    def transfer(self, x0, x1, choices):
        x0, x1 = np.asarray(x0, np.uint8) & 1, np.asarray(x1, np.uint8) & 1
        c = np.asarray(choices, np.uint8)
        if not (x0.shape == x1.shape == c.shape) or x0.ndim != 1 or not len(c):
            raise ValueError('x0, x1, choices must be equal-length nonempty bit vectors')
        q, t, key = self._extend(c)
        y0, y1 = x0 ^ _crh_bits(q, key), x1 ^ _crh_bits(q ^ np.asarray(self._s, np.uint8)[:, None], key)
        return np.where(c == 1, y1, y0) ^ _crh_bits(t, key)

    def transfer_words(self, x0, x1, choices, chunk=1 << 17):
        """Same OT for 64-bit words (uint64 messages); used by Gilboa OLE."""
        x0, x1 = np.asarray(x0, np.uint64), np.asarray(x1, np.uint64)
        c = np.asarray(choices, np.uint8) & 1
        if not (x0.shape == x1.shape == c.shape) or x0.ndim != 1 or not len(c):
            raise ValueError('x0, x1, choices must be equal-length nonempty vectors')
        out = []
        for a in range(0, len(c), chunk):
            s = slice(a, a + chunk)
            q, t, key = self._extend(c[s])
            y0, y1 = x0[s] ^ _crh_words(q, key), x1[s] ^ _crh_words(q ^ np.asarray(self._s, np.uint8)[:, None], key)
            out.append(np.where(c[s] == 1, y1, y0) ^ _crh_words(t, key))
        return np.concatenate(out)


def run_iknp_bits(x0, x1, choices, session: bytes = b''):
    """One-shot batched bit OT (fresh base OTs). Returns x_{c_j}[j] for the receiver."""
    return IKNPBitSession(session).transfer(x0, x1, choices)
