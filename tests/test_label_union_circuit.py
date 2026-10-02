"""Circuit-PSI label union (label_union.circuit_union): grouping, keys, plain == secure."""
import unittest

from label_union.circuit_union import circuit_union_with_keys
from secfl import ristretto as rg


def same_row(index):
    flat = [(c, x, g) for c, d in enumerate(index) for x, g in d.items()]
    return {(a[:2], b[:2]) for a in flat for b in flat if a[2] == b[2]}


class CircuitUnionTests(unittest.TestCase):
    def test_exact_names_split_by_images_keys_and_plain(self):
        labels = [['cat', '3'], ['cat', '3'], ['cat']]
        kws = [{x: x for x in l} for l in labels]
        sets = [{'cat': (0, 1, 2, 3, 4, 5), '3': (10, 11, 12, 13, 14, 15)},
                {'cat': (2, 3, 6, 7, 8, 9), '3': (10, 11, 40, 41, 42, 43)},
                {'cat': (20, 21, 22, 23, 24, 25)}]                             # same name, other pictures
        index, sks, pks, U, st = circuit_union_with_keys(labels, kws, sets, bucket_bits=16)
        plain, _, _, Up, _ = circuit_union_with_keys(labels, kws, sets, secure=False)
        self.assertEqual(same_row(index), same_row(plain))
        self.assertEqual(U, Up)
        self.assertEqual(U, 3)
        self.assertEqual(index[0]['cat'], index[1]['cat'])
        self.assertNotEqual(index[0]['cat'], index[2]['cat'])
        self.assertEqual(index[0]['3'], index[1]['3'])
        self.assertEqual(sks[0]['cat'], sks[1]['cat'])
        self.assertNotEqual(sks[0]['cat'], sks[2]['cat'])
        self.assertEqual(len(pks), U)
        for c, d in enumerate(index):
            for x, g in d.items():
                self.assertEqual((rg.BASE * sks[c][x]).encode(), pks[g])
        self.assertTrue(st['mpc']['estimated'])

    def test_fuzzy_synonyms_merge_others_and_case_split(self):
        labels = [['car', 'truck', 'A'], ['automobile', 'a']]
        kws = [{x: x for x in l} for l in labels]
        try:
            index, _, _, U, _ = circuit_union_with_keys(labels, kws, fuzzy=True, secure=False)
        except RuntimeError as e:                                                  # encoder cache missing
            self.skipTest(str(e))
        self.assertEqual(index[0]['car'], index[1]['automobile'])
        self.assertNotEqual(index[0]['truck'], index[1]['automobile'])
        self.assertNotEqual(index[0]['A'], index[1]['a'])
        self.assertEqual(U, 4)

    def test_own_labels_not_compared_but_join_through_others(self):
        alone = [['car', 'automobile']]
        kws = lambda labels: [{x: x for x in l} for l in labels]
        try:
            _, _, _, U1, _ = circuit_union_with_keys(alone, kws(alone), fuzzy=True, secure=False)
        except RuntimeError as e:                                                  # encoder cache missing
            self.skipTest(str(e))
        both = [['car', 'automobile'], ['car']]
        index, _, _, U2, _ = circuit_union_with_keys(both, kws(both), fuzzy=True, secure=False)
        self.assertEqual(U1, 2)                                                    # one client: never compared
        self.assertEqual(U2, 1)                                                    # via client 1's 'car'
        self.assertEqual(index[0]['car'], index[0]['automobile'])

    def test_mpc_cost_one_step_cheaper(self):
        from label_union.circuit_union import mpc_cost
        one, eight = mpc_cost(6, 62, 384, 45, 1, 1), mpc_cost(6, 62, 384, 45, 8, 1)
        self.assertEqual(one['propagation_steps'], 2)                              # 1 step + 1 check
        self.assertLess(one['mults'], eight['mults'] / 2)


if __name__ == '__main__':
    unittest.main()
