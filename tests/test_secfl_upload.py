import unittest

import numpy as np

from secfl.bb import PublicParams
from secfl.upload import UploadLayout


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.p = PublicParams(labels=('a', 'b', 'c'), frac_bits=12, clip=1.0)
        self.layout = UploadLayout(self.p, {'a': 5, 'b': 3, 'c': 4}, n_max=100, num_clients=4)

    def test_sum_of_encodings_decodes_to_weighted_sums(self):
        rng = np.random.default_rng(0)
        clients = [({'a': rng.normal(size=5) * .1, 'c': rng.normal(size=4) * .1}, {'a': 10, 'c': 7}),
                   ({'a': rng.normal(size=5) * .1}, {'a': 30}),
                   ({'b': -rng.random(3) * .2, 'c': rng.normal(size=4) * .1}, {'b': 5, 'c': 1})]
        total = np.zeros(self.layout.size, np.uint64)
        for g, n in clients:
            total = self.layout.add(total, self.layout.encode(g, n))
        grads, N, G = self.layout.decode(total)
        self.assertEqual(N, {'a': 40, 'b': 5, 'c': 8})
        self.assertEqual(G, {'a': 2, 'b': 1, 'c': 2})
        for L in 'abc':
            want = sum(n[L] * self.layout.clip(g[L]) for g, n in clients if L in g)
            np.testing.assert_allclose(grads[L], want, atol=len(clients) * 2 ** -12)
        self.assertLess(grads['b'].max(), 0)                   # negatives survive mod 2^k

    def test_fixed_length_and_zero_fill(self):
        v = self.layout.encode({'b': np.ones(3) * .1}, {'b': 2})
        self.assertEqual(v.shape, (self.layout.size,))
        self.assertTrue((v[:5] == 0).all() and (v[8:12] == 0).all())

    def test_clip(self):
        g = self.layout.clip(np.ones(5) * 10)
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0)

    def test_overflow_refused(self):
        with self.assertRaises(ValueError):
            UploadLayout(PublicParams(labels=('a',), frac_bits=16), {'a': 2}, n_max=1000, num_clients=50)

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.layout.encode({'a': np.ones(5)}, {'a': 101})         # n > n_max
        with self.assertRaises(ValueError):
            self.layout.encode({'a': np.ones(4)}, {'a': 1})           # wrong size
        with self.assertRaises(ValueError):
            self.layout.encode({'z': np.ones(4)}, {'z': 1})           # unknown label


if __name__ == '__main__':
    unittest.main()
