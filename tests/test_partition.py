import unittest

import numpy as np

from fl_datasets import partition_class_subsets, partition_even


def emnist_like(C=62, seed=0):
    """EMNIST byclass-like imbalance: 10 big digit classes, the rest small."""
    rng = np.random.default_rng(seed)
    sizes = np.r_[np.full(min(10, C), 34000), rng.integers(1900, 9000, max(0, C - 10))]
    return np.repeat(np.arange(C), sizes), np.repeat(np.arange(C), sizes // 6)


class PartitionEvenTests(unittest.TestCase):
    def check(self, y, yt, n, seed, holders=2, max_ratio=None):
        C = int(y.max()) + 1
        own, tr, te = partition_even(y, yt, n, seed, holders)
        self.assertTrue((np.bincount(np.concatenate(own), minlength=C) == holders).all())
        k = [len(m) for m in own]
        self.assertLessEqual(max(k) - min(k), 1)
        for split, labels in ((tr, y), (te, yt)):
            self.assertEqual(sorted(i for v in split.values() for i in v), list(range(len(labels))))
            for i in range(n):
                self.assertTrue(set(labels[split[i]].tolist()) <= set(own[i]))
        if max_ratio:
            loads = np.array([len(tr[i]) for i in range(n)])
            self.assertLess(loads.max() / loads.mean(), max_ratio)

    def test_emnist_10_clients_even(self):
        y, yt = emnist_like()
        self.check(y, yt, 10, seed=0, max_ratio=1.05)

    def test_small_cases_never_get_stuck(self):             # the last classes need distinct free clients
        for n, C, holders in ((2, 3, 2), (3, 5, 2), (4, 7, 3), (5, 11, 2), (10, 13, 2)):
            y, yt = emnist_like(C, seed=C)
            for seed in range(20):
                self.check(y, yt, n, seed, holders)

    def test_full_share_gives_every_holder_all_images_of_its_classes(self):
        y, yt = emnist_like()
        split_own = partition_even(y, yt, 10, 0)[0]
        own, tr, te = partition_even(y, yt, 10, 0, full=True)
        self.assertEqual(own, split_own)                                     # same classes as the split
        for split, labels in ((tr, y), (te, yt)):
            for i in range(10):
                self.assertEqual(sorted(split[i]), np.flatnonzero(np.isin(labels, own[i])).tolist())

    def test_too_few_classes_per_client_is_a_clear_error(self):
        y, yt = emnist_like()
        with self.assertRaisesRegex(ValueError, 'no client'):
            partition_class_subsets(y, yt, 10, 3, 5, seed=0)


if __name__ == '__main__':
    unittest.main()
