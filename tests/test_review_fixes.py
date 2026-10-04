"""Regressions from the accuracy review (one scale per whole generator; subset class names)."""
import unittest

import torch


class ReviewFixTests(unittest.TestCase):
    def test_per_label_row_has_one_scale_block_per_tensor(self):
        from secfl.cbn_gan import DCGANTemplate, PerLabelGenerator, rows
        from secure_cbn import row_spec
        g = PerLabelGenerator(2, DCGANTemplate(8, 3, (8, 4, 2)))
        spec = row_spec(g)
        self.assertEqual([n for n, _, _ in spec], [n for n, _ in g._t[0].named_parameters()])
        self.assertEqual(sum(s[0] for _, s, _ in spec), rows(g).shape[1])

    def test_class_subset_names_are_the_subset_not_all_digits(self):
        from torch.utils.data import Subset, TensorDataset
        from fl_datasets import ClassSubsetDataset
        from setup import label_names
        base = TensorDataset(torch.zeros(4, 1), torch.tensor([2, 6, 2, 6]))
        ds = Subset(ClassSubsetDataset(base, [2, 6], [str(i) for i in range(10)]), [0, 1])
        self.assertEqual(label_names(ds, 'MNIST'), ('2', '6'))


if __name__ == '__main__':
    unittest.main()
