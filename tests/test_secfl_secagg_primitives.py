import itertools
import secrets
import unittest

import numpy as np

from secfl.bb import Identity
from secfl.secagg_primitives import pairwise_seed, prg, shamir_share, shamir_reconstruct


class PrimitiveTests(unittest.TestCase):
    def test_pairwise_seed_symmetric_and_bound(self):
        a, b, c = Identity('a'), Identity('b'), Identity('c')
        s_ab = pairwise_seed(a.exchange(b.public_bytes), 'a', 'b', b'round-1')
        s_ba = pairwise_seed(b.exchange(a.public_bytes), 'b', 'a', b'round-1')
        self.assertEqual(s_ab, s_ba)
        self.assertNotEqual(s_ab, pairwise_seed(a.exchange(c.public_bytes), 'a', 'c', b'round-1'))
        self.assertNotEqual(s_ab, pairwise_seed(a.exchange(b.public_bytes), 'a', 'b', b'round-2'))

    def test_prg_deterministic_range(self):
        seed = secrets.token_bytes(32)
        x, y = prg(seed, 1000), prg(seed, 1000)
        self.assertTrue((x == y).all())
        self.assertLess(int(x.max()), 2 ** 32)
        self.assertGreater(len(np.unique(x)), 990)
        self.assertTrue((prg(seed, 10) == x[:10]).all())          # prefix-stable
        self.assertFalse((prg(secrets.token_bytes(32), 1000) == x).all())
        self.assertEqual(prg(seed, 5, 64).dtype, np.uint64)
        with self.assertRaises(ValueError):
            prg(b'short', 3)

    def test_shamir_any_t_subset(self):
        secret = secrets.token_bytes(32)
        shares = shamir_share(secret, 3, range(1, 6))
        for subset in itertools.combinations(shares, 3):
            self.assertEqual(shamir_reconstruct(list(subset)), secret)
        self.assertEqual(shamir_reconstruct(shares), secret)

    def test_shamir_below_threshold_fails(self):
        secret = secrets.token_bytes(32)
        shares = shamir_share(secret, 3, range(1, 6))
        try:
            self.assertNotEqual(shamir_reconstruct(shares[:2]), secret)
        except ValueError:
            pass                                                   # out of range: also a failure

    def test_shamir_validation(self):
        with self.assertRaises(ValueError):
            shamir_share(b'x', 4, [1, 2, 3])
        with self.assertRaises(ValueError):
            shamir_share(b'x', 2, [1, 1])
        with self.assertRaises(ValueError):
            shamir_reconstruct([(1, 5), (1, 6)])


if __name__ == '__main__':
    unittest.main()
