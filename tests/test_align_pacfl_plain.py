import unittest

import numpy as np
import torch

from align.pacfl_plain import (projection_matrix, projected_basis, similarity, plain_groups,
                               components, original_pacfl, agreement)
from clustering import local_basis


def synthetic_clients(n_groups=3, per_group=3, dim=3 * 32 * 32, rank=4, n=64, seed=0, shared_mean=False):
    """Clients in one group share a mean image and per-label low-rank subspaces."""
    rng = np.random.default_rng(seed)
    clients, truth = [], []
    common = rng.uniform(-1, 1, dim)
    for g in range(n_groups):
        mean = common if shared_mean else rng.uniform(-1, 1, dim)
        spaces = {l: rng.standard_normal((dim, rank)) for l in range(2)}
        for _ in range(per_group):
            ys = np.repeat([0, 1], n // 2)
            xs = np.stack([mean + spaces[y] @ rng.standard_normal(rank) / 3 for y in ys])
            xs = np.tanh(xs + .02 * rng.standard_normal(xs.shape)).reshape(-1, 3, 32, 32)
            clients.append([(torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(ys))])
            truth.append(g)
    return clients, truth


class PacflPlainTests(unittest.TestCase):
    def setUp(self):
        self.clients, self.truth = synthetic_clients()
        self.R = projection_matrix(3 * 32 * 32, 256)
        self.bases = [projected_basis(c, self.R, budget=8) for c in self.clients]

    def test_basis_orthonormal_and_similarity_range(self):
        U = self.bases[0]
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
        self.assertAlmostEqual(similarity(U, U), 1.0, places=6)
        s = similarity(self.bases[0], self.bases[-1])
        self.assertTrue(0 <= s < .5)

    def test_projected_groups_recover_truth_and_match_original_pacfl(self):
        groups, S = plain_groups(self.bases, tau=.5)
        truth = [[i for i, g in enumerate(self.truth) if g == k] for k in range(3)]
        self.assertEqual(groups, truth)
        raw = [local_basis(c, budget=8) for c in self.clients]
        self.assertEqual(agreement(groups, original_pacfl(raw, thresh=20), len(self.clients)), 1.0)

    def test_shared_mean_min_angle_collapses_projected_does_not(self):
        # Every group has the same mean image: PACFL's smallest principal angle sees that shared
        # direction and merges everything; the Frobenius overlap still separates the groups.
        clients, truth = synthetic_clients(shared_mean=True)
        groups, _ = plain_groups([projected_basis(c, self.R, budget=8) for c in clients], tau=.5)
        self.assertEqual(groups, [[i for i, g in enumerate(truth) if g == k] for k in range(3)])
        self.assertEqual(original_pacfl([local_basis(c, budget=8) for c in clients], 20),
                         [list(range(9))])

    def test_components(self):
        A = np.zeros((5, 5), bool)
        A[0, 3] = A[3, 0] = A[1, 2] = A[2, 1] = True
        self.assertEqual(components(A), [[0, 3], [1, 2], [4]])


if __name__ == '__main__':
    unittest.main()
