"""t-out-of-k image matching in the label union (label_union.oprf_union.pairs_union_with_keys)."""
import unittest

import numpy as np

from label_union.domain import ANCHOR_FAMILIES, anchor_set, client_sets
from label_union.oprf_union import pairs_union_with_keys, plain_pairs_grouping
from secfl import ristretto as rg


def same_row(index):
    flat = [(c, x, g) for c, d in enumerate(index) for x, g in d.items()]
    return {(a[:2], b[:2]) for a in flat for b in flat if a[2] == b[2]}


class PairsUnionTests(unittest.TestCase):
    def test_merge_chain_split_and_keys(self):
        keys = [{'cat': 'cat', '3': '3'}, {'cat': 'cat', '3': '3'}, {'cat': 'cat'}, {'kitty': 'cat', 'cat': 'cat'}]
        sets = [{'cat': (0, 1, 2, 3, 4, 5), '3': (10, 11, 12, 13, 14, 15)},
                {'cat': (2, 3, 6, 7, 8, 9), '3': (10, 11, 40, 41, 42, 43)},   # cat: 2 shared with client 0
                {'cat': (20, 21, 22, 23, 24, 25)},                              # cat with other pictures
                {'kitty': (6, 7, 30, 31, 32, 33), 'cat': (6, 7, 30, 31, 32, 34)}]  # chain via client 1; one key
        index, sks, pks, stats = pairs_union_with_keys(keys, sets, t=2, bucket_bits=16)
        plain, U = plain_pairs_grouping(keys, sets, t=2)
        self.assertEqual(same_row(index), same_row(plain))
        self.assertEqual(len(pks), U)
        self.assertEqual(U, 3)                                           # cat (chain), cat (other kind), 3
        self.assertTrue(index[0]['cat'] == index[1]['cat'] == index[3]['kitty'] == index[3]['cat'])
        self.assertNotEqual(index[0]['cat'], index[2]['cat'])
        self.assertEqual(index[0]['3'], index[1]['3'])
        for c, d in enumerate(index):
            for x, g in d.items():
                self.assertEqual((rg.BASE * sks[c][x]).encode(), pks[g])
        self.assertNotEqual(sks[0]['cat'], sks[2]['cat'])
        self.assertEqual(stats['edges'], stats['buckets'] - U)           # a spanning forest: one edge per merge

    def test_one_shared_anchor_is_not_enough(self):
        keys = [{'x': 'x'}, {'x': 'x'}]
        index, U = plain_pairs_grouping(keys, [{'x': (0, 1, 2, 3, 4, 5)}, {'x': (5, 6, 7, 8, 9, 10)}], t=2)
        self.assertEqual(U, 2)


class AnchorSetTests(unittest.TestCase):
    def fresh(self, name, seed, k=16):
        _, gen, kw = next(f for f in ANCHOR_FAMILIES if f[0] == name)
        return gen(k, np.random.default_rng(seed), 32, **kw) * 2 - 1           # pipelines' [-1, 1]

    def test_same_kind_matches_other_kind_does_not(self):
        a, b = anchor_set(self.fresh('strokes/w0.10/n1', 1)), anchor_set(self.fresh('strokes/w0.10/n1', 2))
        p = anchor_set(self.fresh('photo/a1.5', 3))
        self.assertGreaterEqual(len(set(a) & set(b)), 2)
        self.assertLess(len(set(a) & set(p)), 2)

    def test_client_sets_fallback(self):
        sets = client_sets({'3': self.fresh('strokes/w0.10/n1', 4)}, ['3', '7'], k=6)
        self.assertEqual(sets['3'], sets['7'])
        self.assertEqual(len(sets['3']), 6)


if __name__ == '__main__':
    unittest.main()
