import tempfile
import unittest

import numpy as np
import torch

from label_union import mpc_union, index_metrics
from tests.test_long_run_safety import no_cluster, cluster, recount, DICTIONARY

D = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck', '0', '1', '2']


class DictUnionTests(unittest.TestCase):
    def check(self, clients, index, U):
        by_name = {}
        for labels, got in zip(clients, index):
            self.assertEqual(set(got), set(labels))                    # each client: only its own labels
            for x, k in got.items():
                self.assertEqual(by_name.setdefault(x, k), k)           # same label -> same index
        self.assertEqual(U, len({x for c in clients for x in c}))       # Aggregator: |union|
        self.assertEqual(sorted(by_name.values()), list(range(U)))      # distinct labels -> 0..U-1

    def test_union_indices(self):
        clients = [['cat', 'dog'], ['dog', 'ship'], ['ship', 'cat', '0']]
        for shuffle in (True, False):
            index, U, stats = mpc_union(clients, D, workers=2, shuffle=shuffle)
            self.check(clients, index, U)
            self.assertTrue(index_metrics(clients, index, U, D)['exact'])
        index, U, _ = mpc_union(clients, D, shuffle=False)
        self.assertEqual(index[2], {'cat': 0, 'ship': 2, '0': 3})       # unshuffled = dictionary order

    def test_disjoint_identical_and_full_dictionary(self):
        for clients in ([['bird'], ['cat'], ['dog']], [['ship', 'truck']] * 3, [D[:5], D[4:]]):
            index, U, _ = mpc_union(clients, D, workers=2)
            self.check(clients, index, U)

    def test_shuffled_order_is_random(self):
        clients = [D[:5], D[5:]]
        orders = {tuple(mpc_union(clients, D)[0][0][x] for x in D[:5]) for _ in range(4)}
        self.assertGreater(len(orders), 1)                               # index != dictionary rank

    def test_index_metrics_catch_errors(self):
        clients = [['cat', 'dog'], ['dog', 'ship']]
        good = [{'cat': 0, 'dog': 1}, {'dog': 1, 'ship': 2}]
        self.assertTrue(index_metrics(clients, good, 3, D)['exact'])
        split = index_metrics(clients, [{'cat': 0, 'dog': 1}, {'dog': 2, 'ship': 0}], 3, D)
        self.assertEqual((split['split_labels'], split['merged_indices'], split['exact']), (['dog'], [0], False))
        self.assertLess(split['mcc'], 1)
        self.assertEqual(index_metrics(clients, good, 4, D)['unused_indices'], [3])
        self.assertFalse(index_metrics(clients, good, 4, D)['exact'])     # wrong U
        self.assertEqual(index_metrics(clients, good, 2, D)['out_of_range'], [2])


class MpcUnionPipelineTests(unittest.TestCase):
    def test_no_cluster_aggregator_gets_only_indices(self):
        result, spaces, tests = no_cluster(union='mpc', rounds=2)
        setup = result['setup']
        self.assertEqual(setup['union']['method'], 'mpc')
        self.assertEqual(setup['labels'], 3)
        self.assertNotIn('holders', setup['union'])                     # no counts, no names
        self.assertTrue(setup['union_metrics']['exact'])
        self.assertEqual(setup['union_metrics']['mcc'], 1.0)
        # independent recount: class k's name, from the experimenter view
        names = {r['cls']: r['label'] for r in setup['experimenter_view']}
        self.assertEqual(sorted(names.values()), ['cat', 'dog', 'ship'])
        existing = [names[k] for k in range(3)]
        self.assertAlmostEqual(result['history'][-1]['accuracy'], recount(result['model'], spaces, tests, existing))
        self.assertEqual(result['history'][-1]['evaluation']['ambiguous_predictions'], 0)

    def test_no_cluster_resume_reuses_the_union(self):
        with tempfile.TemporaryDirectory() as d:
            full, _, _ = no_cluster(union='mpc', rounds=3, checkpoint_dir=d)
            resumed, _, _ = no_cluster(union='mpc', rounds=3, resume=f'{d}/checkpoint_last.pt')
            with self.assertRaises(ValueError):                           # other union method
                no_cluster(union='secagg', rounds=4, resume=f'{d}/checkpoint_last.pt')
        self.assertEqual(resumed['setup']['union'], full['setup']['union'])   # not recomputed
        np.testing.assert_array_equal(resumed['theta'], full['theta'])

    def test_cluster_groups_over_indices(self):
        result, spaces, tests = cluster(union='mpc', cluster='private', workers=2)
        setup = result['setup']
        self.assertEqual(setup['groups'], 2)
        held = setup['aggregator_view']
        self.assertTrue(all(isinstance(k, int) for v in held.values() for k in v))   # indices only
        self.assertEqual(sorted(k for v in held.values() for k in v), [0, 1, 2, 3])
        self.assertEqual(setup['union_metrics']['mcc'], 1.0)
        names = {r['cls']: r['label'] for r in setup['union_view']}
        self.assertAlmostEqual(result['history'][-1]['accuracy'],
                               recount(result['model'], spaces, tests, [names[k] for k in range(4)]))


if __name__ == '__main__':
    unittest.main()
