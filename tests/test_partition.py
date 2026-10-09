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


class LabelSplitTests(unittest.TestCase):
    def test_full_share_allows_classes_nobody_drew(self):            # 6 clients x 3-4 of 62 classes
        y, yt = emnist_like()
        own, tr, te = partition_class_subsets(y, yt, 6, 3, 4, seed=0, full=True)
        self.assertTrue(all(3 <= len(o) <= 4 for o in own))
        cover = np.bincount(np.concatenate(own), minlength=62)
        self.assertLessEqual(cover.max() - cover[cover > 0].min(), 1)   # even: least covered first
        for i in range(6):
            self.assertEqual(sorted(tr[i]), np.flatnonzero(np.isin(y, own[i])).tolist())

    def test_two_holders_per_label_counting_special_clients(self):   # 50-client case: lo * clients >= need
        y, yt = emnist_like()
        for seed in range(20):
            own, _, _ = partition_class_subsets(y, yt, 30, 5, 6, seed=seed, full=True)
            self.assertGreaterEqual(np.bincount(np.concatenate(own), minlength=62).min(), 2)
        y10 = np.arange(1000) % 10
        cover0 = [1] * 9 + [0]                                         # specials hold the 9 shared names once
        for seed in range(20):
            own, _, _ = partition_class_subsets(y10, y10, 3, 5, 6, seed=seed, full=True, cover0=cover0)
            self.assertGreaterEqual((np.bincount(np.concatenate(own), minlength=10) + cover0).min(), 2)

    def test_cifar_stl_clients_merge_both_datasets_per_name(self):
        from unittest.mock import patch
        import torch
        from torch.utils.data import TensorDataset
        import fl_datasets
        from setup import label_names

        def fake(name, root, train):
            names = fl_datasets.stl10_classes() if name == 'STL10' else CIFAR
            ds = TensorDataset(torch.full((40, 1), 1. if name == 'STL10' else 0.), torch.arange(40) % 10)
            ds.targets = (torch.arange(40) % 10).tolist()
            return ds
        CIFAR = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        with patch.object(fl_datasets, 'get_raw_dataset_transform', fake), \
                patch.object(fl_datasets, 'get_readable_class_names',
                             lambda n, root=None: fl_datasets.stl10_classes() if n == 'STL10' else CIFAR):
            entries = fl_datasets.mixed_cifar_stl_clients(4, 3, 4, 0, '/x', 8)
        for e in entries:
            names = label_names(e['train'].dataset, 'CIFAR10+STL10')
            self.assertTrue(3 <= len(names) <= 4 and not {'frog', 'monkey'} & set(names))
            seen = {}
            for x, lab in e['train'].dataset:                         # each local label: 4 CIFAR + 4 STL images
                seen.setdefault(int(lab), []).append(float(x[0]))
            self.assertEqual(sorted(seen), list(range(len(names))))
            self.assertTrue(all(sorted(v) == [0.] * 4 + [1.] * 4 for v in seen.values()))
            self.assertEqual([d for d, _ in e['tests']], ['CIFAR10', 'STL10'])
            for d, loader in e['tests']:                               # per-dataset test views, same labels
                self.assertEqual(label_names(loader.dataset, d), names)


if __name__ == '__main__':
    unittest.main()
