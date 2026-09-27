"""A4a: n-party Boolean MPC (GMW), semi-honest, secure against any n-1 colluding parties.

Every value is XOR-shared among all n parties (party 0 also absorbs public constants).
XOR/NOT are local. AND uses a Beaver triple (a, b, c = a&b) with
  c = XOR_i a_i b_i  XOR  XOR_{i != j} a_i b_j,
where each cross term a_i b_j comes from one bit-OT between i (chooser, bit a_i) and
j (sender, messages r, r^b_j) over a persistent IKNP session per ordered pair.
Triples are made in large batches (pool) so a circuit layer costs one open() round.

ponytail: n(n-1) base-OT setups at ~1 s each in pure Python (ristretto255); fine for
tests with a few parties, use libsodium or an MPC framework (MP-SPDZ) for 30 clients.
"""
import secrets

import numpy as np

from secfl.ot import IKNPBitSession


def rand_bits(shape):
    n = int(np.prod(shape))
    raw = np.frombuffer(secrets.token_bytes((n + 7) // 8), np.uint8)
    return np.unpackbits(raw)[:n].reshape(shape)


class Shared:
    """XOR shares of a bit array: parts[i] is party i's share."""
    __slots__ = ('parts',)

    def __init__(self, parts):
        self.parts = [np.asarray(p, np.uint8) for p in parts]

    @property
    def shape(self):
        return self.parts[0].shape

    def __xor__(self, o):
        if isinstance(o, Shared):
            return Shared([a ^ b for a, b in zip(self.parts, o.parts)])
        c = np.asarray(o, np.uint8)
        return Shared([self.parts[0] ^ c] + self.parts[1:])

    def __invert__(self):
        return self ^ 1

    def __getitem__(self, idx):
        return Shared([p[idx] for p in self.parts])

    def reshape(self, *shape):
        return Shared([p.reshape(*shape) for p in self.parts])

    def broadcast_to(self, shape):
        return Shared([np.broadcast_to(p, shape) for p in self.parts])

    def xor_reduce(self, axis):
        return Shared([np.bitwise_xor.reduce(p, axis=axis) for p in self.parts])

    def set(self, idx, value: 'Shared'):
        """Functional update: copy with self[idx] = value."""
        parts = [p.copy() for p in self.parts]
        for p, v in zip(parts, value.parts):
            p[idx] = v
        return Shared(parts)


def concat(items, axis=-1):
    return Shared([np.concatenate([s.parts[i] for s in items], axis=axis) for i in range(len(items[0].parts))])


class MPC:
    def __init__(self, n_parties: int, session: bytes = b'', chunk: int = 1 << 16):
        if n_parties < 2:
            raise ValueError('need at least 2 parties')
        self.n, self.session, self.chunk = n_parties, session, chunk
        self.and_gates = self.rounds = 0
        self._ot = {}                                   # (chooser i, sender j) -> session
        self._pool = None
        self._used = 0

    def _session(self, i, j):
        if (i, j) not in self._ot:
            self._ot[i, j] = IKNPBitSession(self.session + f'/ot/{i}<-{j}'.encode())
        return self._ot[i, j]

    def warm(self, workers=1):
        """All n(n-1) base-OT setups up front, in processes: in pure Python they dominate setup."""
        keys = [(i, j) for i in range(self.n) for j in range(self.n) if i != j and (i, j) not in self._ot]
        names = [self.session + f'/ot/{i}<-{j}'.encode() for i, j in keys]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as pool:
                sessions = list(pool.map(IKNPBitSession, names))
        else:
            sessions = list(map(IKNPBitSession, names))
        self._ot.update(zip(keys, sessions))
        return self

    # ---------------------------------------------------------------- inputs/outputs
    def input(self, bits, owner: int) -> Shared:
        """Owner splits its bits: one random share to every other party."""
        bits = np.asarray(bits, np.uint8) & 1
        parts = [rand_bits(bits.shape) for _ in range(self.n)]
        acc = np.zeros_like(bits)
        for i, p in enumerate(parts):
            if i != owner:
                acc ^= p
        parts[owner] = bits ^ acc
        return Shared(parts)

    def constant(self, bits) -> Shared:
        bits = np.asarray(bits, np.uint8) & 1
        return Shared([bits] + [np.zeros_like(bits) for _ in range(self.n - 1)])

    def open(self, x: Shared):
        """Every party broadcasts its share."""
        out = np.zeros(x.shape, np.uint8)
        for p in x.parts:
            out = out ^ p
        return out

    def reveal_to(self, x: Shared, party: int):
        """All other parties send their shares to `party` only."""
        return self.open(x)

    # ---------------------------------------------------------------- triples
    def _make_triples(self, count):
        a = [rand_bits((count,)) for _ in range(self.n)]
        b = [rand_bits((count,)) for _ in range(self.n)]
        c = [a[i] & b[i] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    r = rand_bits((count,))                                 # sender j's mask
                    got = self._session(i, j).transfer(r, r ^ b[j], a[i])   # = r ^ a_i b_j
                    c[i] ^= got
                    c[j] ^= r
        return a, b, c

    def _take(self, count):
        if self._pool is None or self._used + count > len(self._pool[0][0]):
            self._pool, self._used = self._make_triples(max(count, self.chunk)), 0
        s = slice(self._used, self._used + count)
        self._used += count
        return [Shared([p[s] for p in part]) for part in self._pool]

    # ---------------------------------------------------------------- gates
    def and_(self, x: Shared, y: Shared) -> Shared:
        if x.shape != y.shape:
            shape = np.broadcast_shapes(x.shape, y.shape)
            x, y = x.broadcast_to(shape), y.broadcast_to(shape)
        shape, count = x.shape, int(np.prod(x.shape))
        if count == 0:
            return Shared([np.zeros(shape, np.uint8) for _ in range(self.n)])
        a, b, c = (t.reshape(*shape) for t in self._take(count))
        d, e = self.open(x ^ a), self.open(y ^ b)                   # masked, uniform
        self.and_gates += count
        self.rounds += 1
        parts = [c.parts[i] ^ (d & b.parts[i]) ^ (e & a.parts[i]) for i in range(self.n)]
        parts[0] = parts[0] ^ (d & e)
        return Shared(parts)
