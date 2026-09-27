import unittest

import numpy as np

from align.twopc import TwoPC
from secfl.ot import run_iknp_bits


class BitOTTests(unittest.TestCase):
    def test_bit_ot(self):
        rng = np.random.default_rng(0)
        x0, x1, c = (rng.integers(0, 2, 5000, dtype=np.uint8) for _ in range(3))
        np.testing.assert_array_equal(run_iknp_bits(x0, x1, c), np.where(c == 1, x1, x0))


class TwoPCTests(unittest.TestCase):
    def setUp(self):
        self.pc, self.rng = TwoPC(), np.random.default_rng(1)

    def bits(self, *shape):
        return self.rng.integers(0, 2, shape, dtype=np.uint8)

    def test_shares_look_random(self):
        x = self.pc.input(np.zeros(4000, np.uint8), owner=0)
        self.assertAlmostEqual(x.s1.mean(), .5, delta=.05)          # party 1's view is uniform

    def test_xor_not_and(self):
        a, b = self.bits(50, 7), self.bits(50, 7)
        x, y = self.pc.input(a, 0), self.pc.input(b, 1)
        np.testing.assert_array_equal(self.pc.open(x ^ y), a ^ b)
        np.testing.assert_array_equal(self.pc.open(~x), 1 - a)
        np.testing.assert_array_equal(self.pc.open(self.pc.and_(x, y)), a & b)
        np.testing.assert_array_equal(self.pc.open(self.pc.or_(x, y)), a | b)

    def test_eq_and_reduce(self):
        a = self.bits(30, 41)
        b = a.copy(); b[::2, 5] ^= 1                                  # even rows differ
        e = self.pc.eq(self.pc.input(a, 0), self.pc.input(b, 1))
        np.testing.assert_array_equal(self.pc.open(e), np.arange(30) % 2)

    def test_mux(self):
        bit, u, v = self.bits(20), self.bits(20, 16), self.bits(20, 16)
        out = self.pc.mux(self.pc.input(bit, 0), self.pc.input(u, 1), self.pc.input(v, 1))
        np.testing.assert_array_equal(self.pc.open(out), np.where(bit[:, None] == 1, u, v))


if __name__ == '__main__':
    unittest.main()
