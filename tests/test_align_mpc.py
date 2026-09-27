import unittest

import numpy as np

from align.mpc import MPC, concat

ENGINE = MPC(3, b'test-mpc')


class MPCTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def bits(self, *shape):
        return self.rng.integers(0, 2, shape, dtype=np.uint8)

    def test_input_open_xor_not(self):
        a, b = self.bits(40), self.bits(40)
        x, y = ENGINE.input(a, 0), ENGINE.input(b, 2)
        self.assertEqual(len(x.parts), 3)
        np.testing.assert_array_equal(ENGINE.open(x ^ y), a ^ b)
        np.testing.assert_array_equal(ENGINE.open(~x), 1 - a)
        np.testing.assert_array_equal(ENGINE.open(x ^ 1), 1 - a)

    def test_and_many(self):
        a, b = self.bits(100, 9), self.bits(100, 9)
        z = ENGINE.and_(ENGINE.input(a, 1), ENGINE.input(b, 2))
        np.testing.assert_array_equal(ENGINE.open(z), a & b)

    def test_and_with_constant_and_broadcast(self):
        a = self.bits(5, 4)
        z = ENGINE.and_(ENGINE.input(a, 0), ENGINE.constant(np.array([1, 0, 1, 0], np.uint8)))
        np.testing.assert_array_equal(ENGINE.open(z), a & [1, 0, 1, 0])

    def test_any_n_minus_1_shares_uniform(self):
        x = ENGINE.input(np.zeros(20000, np.uint8), owner=0)
        for p in x.parts[1:]:
            self.assertAlmostEqual(float(p.mean()), .5, delta=.02)
        self.assertAlmostEqual(float((x.parts[1] ^ x.parts[2]).mean()), .5, delta=.02)

    def test_set_and_concat(self):
        a = self.bits(6, 3)
        x = ENGINE.input(a, 0)
        y = x.set(slice(0, 2), ENGINE.constant(np.ones((2, 3), np.uint8)))
        want = a.copy(); want[:2] = 1
        np.testing.assert_array_equal(ENGINE.open(y), want)
        np.testing.assert_array_equal(ENGINE.open(concat([x, y], axis=1)), np.concatenate([a, want], 1))


if __name__ == '__main__':
    unittest.main()
