import unittest

import numpy as np

from align.mpc import MPC, concat
from align.circuits import eq, lt, cond_swap, bitonic_sort, add, prefix_sum, to_bits, from_bits

E = MPC(3, b'test-circuits')


class CircuitTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_bits_roundtrip(self):
        v = np.array([0, 5, 1023])
        np.testing.assert_array_equal(from_bits(to_bits(v, 10)), v)
        np.testing.assert_array_equal(from_bits(to_bits(v, 10, False), False), v)

    def test_eq_lt(self):
        a = self.rng.integers(0, 64, 200); b = self.rng.integers(0, 64, 200); b[:40] = a[:40]
        for width in (6, 7):                                       # 7: non power of two
            A, B = E.input(to_bits(a, width), 0), E.input(to_bits(b, width), 1)
            np.testing.assert_array_equal(E.open(eq(E, A, B)), (a == b).astype(int))
            np.testing.assert_array_equal(E.open(lt(E, A, B)), (a < b).astype(int))

    def test_cond_swap(self):
        x, y, s = self.rng.integers(0, 2, (10, 5)), self.rng.integers(0, 2, (10, 5)), self.rng.integers(0, 2, 10)
        X, Y = cond_swap(E, E.input(s, 2), E.input(x, 0), E.input(y, 1))
        np.testing.assert_array_equal(E.open(X), np.where(s[:, None] == 1, y, x))
        np.testing.assert_array_equal(E.open(Y), np.where(s[:, None] == 1, x, y))

    def test_bitonic_sort_carries_payload(self):
        keys = self.rng.integers(0, 256, 16); keys[3] = keys[9]    # with a duplicate
        payload = np.arange(16)
        rec = concat([E.input(to_bits(keys, 8), 0), E.constant(to_bits(payload, 4))])
        out = E.open(bitonic_sort(E, rec, 8))
        k, p = from_bits(out[:, :8]), from_bits(out[:, 8:])
        np.testing.assert_array_equal(k, np.sort(keys))
        np.testing.assert_array_equal(keys[p], k)                   # payload travelled with key

    def test_add_prefix_sum(self):
        a, b = self.rng.integers(0, 100, 30), self.rng.integers(0, 100, 30)
        s = add(E, E.input(to_bits(a, 8, False), 0), E.input(to_bits(b, 8, False), 1))
        np.testing.assert_array_equal(from_bits(E.open(s), False), (a + b) % 256)
        f = self.rng.integers(0, 2, 13)
        ps = prefix_sum(E, E.input(to_bits(f, 4, False), 2))
        np.testing.assert_array_equal(from_bits(E.open(ps), False), np.cumsum(f))


if __name__ == '__main__':
    unittest.main()
