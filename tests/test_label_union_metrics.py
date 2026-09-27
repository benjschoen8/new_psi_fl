import unittest

from label_union import union_metrics, true_union

D = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']


class UnionMetricsTests(unittest.TestCase):
    def test_exact(self):
        m = union_metrics(['cat', 'dog'], {'dog', 'cat'}, D)
        self.assertEqual((m['true_positive'], m['false_positive'], m['false_negative'], m['true_negative']), (2, 0, 0, 4))
        self.assertEqual((m['mcc'], m['f1'], m['jaccard'], m['exact']), (1.0, 1.0, 1.0, True))

    def test_missing_label(self):
        m = union_metrics(['cat'], ['cat', 'dog'], D)
        self.assertEqual((m['false_negative'], m['missing'], m['recall']), (1, ['dog'], .5))
        self.assertAlmostEqual(m['mcc'], (1 * 4 - 0) / (1 * 2 * 4 * 5) ** .5)
        self.assertFalse(m['exact'])

    def test_spurious_label(self):
        m = union_metrics(['cat', 'dog', 'ship'], ['cat', 'dog'], D)
        self.assertEqual((m['false_positive'], m['spurious'], m['precision']), (1, ['ship'], 2 / 3))
        self.assertLess(m['mcc'], 1)

    def test_dictionary_is_the_union(self):
        # TN = 0 makes the MCC formula 0/0: exact -> 1, any error -> 0
        self.assertEqual(union_metrics(D, D, D)['mcc'], 1.0)
        self.assertEqual(union_metrics(D[:-1], D, D)['mcc'], 0.0)

    def test_holder_counts(self):
        real, holders = true_union([['cat', 'dog'], ['dog'], ['dog', 'dog']])
        self.assertEqual((real, holders), ({'cat', 'dog'}, {'cat': 1, 'dog': 3}))
        self.assertTrue(union_metrics(real, real, D, {'cat': 1, 'dog': 3}, holders)['holders_exact'])
        self.assertFalse(union_metrics(real, real, D, {'cat': 1, 'dog': 2}, holders)['holders_exact'])

    def test_label_outside_dictionary(self):
        with self.assertRaises(ValueError):
            union_metrics(['kitten'], ['cat'], D)


if __name__ == '__main__':
    unittest.main()
