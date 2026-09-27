import unittest

import numpy as np

from secfl.label_codes import label_code
from label_union import indicator, discover

DICT = ('bird', 'cat', 'dog', 'horse', 'ship', 'truck')


class LabelCodeTests(unittest.TestCase):
    def test_public_codes_agree(self):
        np.testing.assert_array_equal(label_code('cat', 64), label_code('cat', 64))
        self.assertLess(abs(float(label_code('cat', 128) @ label_code('dog', 128))), .4)

    def test_discover_existence_and_counts(self):
        existing, counts, stats = discover([['cat', 'dog'], ['dog', 'ship'], ['ship', 'cat', 'bird']], DICT)
        self.assertEqual(existing, ['bird', 'cat', 'dog', 'ship'])           # dictionary order
        self.assertEqual(counts, [1, 2, 2, 2])
        self.assertEqual(stats['upload_bytes_per_client'], len(DICT) * 8)

    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            indicator(['kitten'], DICT)


if __name__ == '__main__':
    unittest.main()
