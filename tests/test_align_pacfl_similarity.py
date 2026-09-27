import unittest

import numpy as np

from align.pacfl_plain import projection_matrix, projected_basis, similarity
from align.pacfl_similarity import pair_vectors, ole_inner, similar_bit, all_pairs, FRAC
from secfl.ot import IKNPBitSession
from tests.test_align_pacfl_plain import synthetic_clients


def rand_basis(rng, r, p):
    return np.linalg.qr(rng.standard_normal((r, p)))[0]


class PacflSimilarityTests(unittest.TestCase):
    def test_ole_inner_product(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 1 << 10, 500).astype(np.uint64)
        b = rng.integers(-(1 << 20), 1 << 20, 500)
        zi, zj = ole_inner(a, b.astype(np.uint64), IKNPBitSession(b't'), 10)
        with np.errstate(over='ignore'):
            got = int(np.array([zi + zj], np.uint64).view(np.int64)[0])
        self.assertEqual(got, int((a.astype(np.int64) * b).sum()))

    def test_fixed_point_inner_product_matches_similarity(self):
        rng = np.random.default_rng(2)
        Ui, Uj = rand_basis(rng, 32, 5), rand_basis(rng, 32, 7)
        a, _ = pair_vectors(Ui)
        _, b = pair_vectors(Uj)
        approx = ((a.astype(np.int64) - (1 << FRAC)) * b.astype(np.int64)).sum() / 2 ** (2 * FRAC)
        self.assertAlmostEqual(approx, np.linalg.norm(Ui.T @ Uj) ** 2, places=2)

    def test_bit_matches_plain_threshold(self):
        rng = np.random.default_rng(3)
        U = rand_basis(rng, 24, 4)
        near = np.linalg.qr(U + .05 * rng.standard_normal(U.shape))[0]
        far = rand_basis(rng, 24, 6)
        for V in (U, near, far):
            s = similarity(U, V)
            for tau in (.3, .9):
                ei, ej, _ = similar_bit(U, V, tau, b'pair')
                self.assertEqual(ei ^ ej, int(s > tau), (s, tau))

    def test_all_pairs_recovers_plain_adjacency(self):
        clients, _ = synthetic_clients()
        R = projection_matrix(3 * 32 * 32, 32)
        bases = [projected_basis(c, R, budget=8) for c in clients]
        E, stats = all_pairs(bases, .5, workers=4)
        A = E ^ E.T
        n = len(bases)
        want = np.array([[i != j and similarity(bases[i], bases[j]) > .5 for j in range(n)] for i in range(n)])
        np.testing.assert_array_equal(A.astype(bool), want)
        self.assertEqual(stats['pairs'], n * (n - 1) // 2)
        # shares alone look random: a client's row says nothing without the partner's share
        self.assertTrue(0 < E.sum() < n * n)


if __name__ == '__main__':
    unittest.main()
