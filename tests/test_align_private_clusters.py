import unittest

import numpy as np

from align.private_clusters import private_clusters, plain_reference


def share(A, rng):
    """Split a symmetric adjacency into P2-style pairwise XOR shares."""
    n = len(A)
    E = np.zeros((n, n), np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            E[i, j] = rng.integers(0, 2)
            E[j, i] = E[i, j] ^ int(A[i, j])
    return E


class PrivateClustersTests(unittest.TestCase):
    def check(self, A, ind, seed=0):
        rng = np.random.default_rng(seed)
        got = private_clusters(share(A, rng), ind, workers=4)
        want = plain_reference(A, ind)
        self.assertEqual(got['groups'], want['groups'])
        self.assertEqual(got['clients'], want['clients'])
        np.testing.assert_array_equal(got['labels'], want['labels'])
        return got

    def test_chain_needs_transitive_closure(self):
        # 0-3, 3-1 (chain: 0 and 1 are not directly similar), 2 alone, 4-5
        n = 6
        A = np.zeros((n, n), bool)
        for i, j in [(0, 3), (3, 1), (4, 5)]:
            A[i, j] = A[j, i] = True
        ind = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        got = self.check(A, ind)
        self.assertEqual(got['clients'][1]['members'], [0, 1, 3])
        self.assertEqual([c['slot'] for c in got['clients']], [0, 0, 1, 0, 2, 2])
        # server view: only per-group existence bits, no sizes (group {4, 5} both hold label 3 -> still 1)
        np.testing.assert_array_equal(got['labels'].astype(int), [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

    def test_all_alone_and_all_together(self):
        n = 4
        ind = np.eye(n, dtype=np.uint8)
        self.assertEqual(self.check(np.zeros((n, n), bool), ind)['groups'], 4)
        self.assertEqual(self.check(~np.eye(n, dtype=bool), ind, seed=1)['groups'], 1)


if __name__ == '__main__':
    unittest.main()
