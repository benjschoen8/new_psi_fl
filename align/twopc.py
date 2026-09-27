"""A2: two-party Boolean circuits (GMW), semi-honest. Party 0 and party 1 hold XOR shares.

XOR/NOT are local; AND consumes a Beaver triple (a, b, c = a&b) produced from two bit-OTs
(IKNP, secfl.ot.run_iknp_bits): a0&b1 and a1&b0 are the only cross terms.
Shares are numpy uint8 arrays of any shape, so a whole layer of gates runs as one batch.
Both parties are simulated in one process, but every value crossing between them goes
through an explicit OT or an open() of masked values.
"""
import secrets

import numpy as np

from secfl.ot import IKNPBitSession


def _rand_bits(shape):
    n = int(np.prod(shape))
    raw = np.frombuffer(secrets.token_bytes((n + 7) // 8), np.uint8)
    return np.unpackbits(raw)[:n].reshape(shape)


class Shared:
    __slots__ = ('s0', 's1')

    def __init__(self, s0, s1):
        self.s0, self.s1 = np.asarray(s0, np.uint8), np.asarray(s1, np.uint8)

    @property
    def shape(self):
        return self.s0.shape

    def __xor__(self, o):
        if isinstance(o, Shared):
            return Shared(self.s0 ^ o.s0, self.s1 ^ o.s1)
        return Shared(self.s0 ^ np.asarray(o, np.uint8), self.s1)       # public constant: party 0 adds it

    def __invert__(self):
        return self ^ 1

    def __getitem__(self, idx):
        return Shared(self.s0[idx], self.s1[idx])

    def reshape(self, *shape):
        return Shared(self.s0.reshape(*shape), self.s1.reshape(*shape))

    def xor_reduce(self, axis):
        return Shared(np.bitwise_xor.reduce(self.s0, axis=axis), np.bitwise_xor.reduce(self.s1, axis=axis))

    def broadcast_to(self, shape):
        return Shared(np.broadcast_to(self.s0, shape), np.broadcast_to(self.s1, shape))


class TwoPC:
    def __init__(self, session: bytes = b''):
        self.session = session
        self.and_gates = self.rounds = 0
        self._ot10 = IKNPBitSession(session + b'/1->0')     # party 1 sends, party 0 chooses
        self._ot01 = IKNPBitSession(session + b'/0->1')     # party 0 sends, party 1 chooses

    # inputs / outputs
    def input(self, bits, owner: int) -> Shared:
        bits = np.asarray(bits, np.uint8) & 1
        r = _rand_bits(bits.shape)
        return Shared(bits ^ r, r) if owner == 0 else Shared(r, bits ^ r)

    def constant(self, bits) -> Shared:
        bits = np.asarray(bits, np.uint8) & 1
        return Shared(bits, np.zeros_like(bits))

    def open(self, x: Shared):
        return x.s0 ^ x.s1

    def reveal_to(self, x: Shared, party: int):
        """The other party sends its share; only `party` learns the value."""
        return x.s0 ^ x.s1

    # gates
    def _triples(self, shape):
        n = int(np.prod(shape))
        a0, b0, a1, b1 = (_rand_bits((n,)) for _ in range(4))
        r, s = _rand_bits((n,)), _rand_bits((n,))
        u0 = self._ot10.transfer(r, r ^ b1, a0)                    # party 1 sends, party 0 picks a0
        v1 = self._ot01.transfer(s, s ^ b0, a1)                    # party 0 sends, party 1 picks a1
        c0 = (a0 & b0) ^ u0 ^ s                                     # u0 ^ r = a0 b1 ; s ^ v1 = a1 b0
        c1 = (a1 & b1) ^ r ^ v1
        f = lambda v: v.reshape(shape)
        return Shared(f(a0), f(a1)), Shared(f(b0), f(b1)), Shared(f(c0), f(c1))

    def and_(self, x: Shared, y: Shared) -> Shared:
        if x.shape != y.shape:
            x, y = x.broadcast_to(np.broadcast_shapes(x.shape, y.shape)), y.broadcast_to(np.broadcast_shapes(x.shape, y.shape))
        if x.s0.size == 0:
            return x
        a, b, c = self._triples(x.shape)
        d, e = self.open(x ^ a), self.open(y ^ b)                   # uniformly masked
        self.and_gates += x.s0.size
        self.rounds += 1
        return Shared(c.s0 ^ (d & b.s0) ^ (e & a.s0) ^ (d & e), c.s1 ^ (d & b.s1) ^ (e & a.s1))

    # gadgets
    def and_reduce(self, x: Shared, axis=-1) -> Shared:
        """AND over one axis with a log-depth tree."""
        x = Shared(np.moveaxis(x.s0, axis, -1), np.moveaxis(x.s1, axis, -1))
        while x.shape[-1] > 1:
            if x.shape[-1] % 2:
                one = self.constant(np.ones(x.shape[:-1] + (1,), np.uint8))
                x = Shared(np.concatenate([x.s0, one.s0], -1), np.concatenate([x.s1, one.s1], -1))
            h = x.shape[-1] // 2
            x = self.and_(x[..., :h], x[..., h:])
        return x[..., 0]

    def or_(self, x, y):
        return ~self.and_(~x, ~y)

    def eq(self, x: Shared, y: Shared, axis=-1) -> Shared:
        return self.and_reduce(~(x ^ y), axis)

    def mux(self, bit: Shared, if1: Shared, if0: Shared) -> Shared:
        """bit ? if1 : if0, with bit broadcast over the trailing axis of the words."""
        if bit.s0.ndim < if1.s0.ndim:
            bit = bit.reshape(*bit.shape, *([1] * (if1.s0.ndim - bit.s0.ndim)))
        b = bit.broadcast_to(if1.shape) if bit.shape != if1.shape else bit
        return if0 ^ self.and_(b, if1 ^ if0)
