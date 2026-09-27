import unittest


class ProtocolTests(unittest.TestCase):
    def test_new_modules_exist(self):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec('main'))

    def test_by_class_preserves_permutations_and_case(self):
        from mapping import ByClassMapping
        truth = ByClassMapping({'a': ('0', 'A', 'a'), 'b': ('a', '0')})(None)
        self.assertEqual(truth['a'][0], truth['b'][1])
        self.assertEqual(truth['a'][2], truth['b'][0])
        self.assertNotEqual(truth['a'][1], truth['a'][2])

    def test_wrong_merge_is_not_perfect_accuracy(self):
        from evaluation import semantic_alignment, mapping_metrics
        truth = {'a': {0: 0, 1: 1}, 'b': {0: 0, 1: 1}}
        predicted = {'a': {0: 7, 1: 7}, 'b': {0: 7, 1: 7}}
        self.assertEqual(semantic_alignment(predicted, truth), {})
        self.assertEqual(mapping_metrics(predicted, truth)['false_positive'], 2)

    def test_mapping_metrics_preserve_legacy_formulas(self):
        from label_mapping_utils import evaluate_mapping_results
        from mapping import ByClassMapping
        from evaluation import mapping_metrics
        spaces = {'a': ('a', 'b', 'c'), 'b': ('a', 'b', 'd')}
        predicted = {'a': {0: 0, 1: 1, 2: 2}, 'b': {0: 0, 1: 2, 2: 3}}
        old = evaluate_mapping_results(list(spaces), spaces, predicted)
        new = mapping_metrics(predicted, ByClassMapping(spaces)())
        for current, legacy in {'true_positive': 'TP', 'false_positive': 'FP',
                'true_negative': 'TN', 'false_negative': 'FN', 'precision': 'Precision',
                'recall': 'Recall', 'specificity': 'Specificity', 'f1': 'F1-Score',
                'balanced_accuracy': 'AvgAccuracy', 'mcc': 'MCC'}.items():
            self.assertAlmostEqual(new[current], old[legacy], msg=current)
        self.assertAlmostEqual(new['pair_accuracy'], 7 / 9)
        self.assertAlmostEqual(new['balanced_accuracy'], (.5 + 6 / 7) / 2)
        self.assertAlmostEqual(new['mcc'], 5 / 14)

    def test_mapping_mcc_degenerate_denominator_matches_legacy(self):
        from evaluation import mapping_metrics
        for truth, prediction in [({'a': {0: 0}, 'b': {0: 0}}, {'a': {0: 0}, 'b': {0: 0}}),
                                  ({'a': {0: 0}}, {'a': {0: 0}})]:
            self.assertEqual(mapping_metrics(prediction, truth)['mcc'], 0.)

    def test_global_ids_do_not_need_to_match(self):
        from evaluation import semantic_alignment
        truth = {'a': {0: 0, 1: 1}, 'b': {0: 1}}
        predicted = {'a': {0: 8, 1: 3}, 'b': {0: 3}}
        self.assertEqual(semantic_alignment(predicted, truth), {8: 0, 3: 1})

    def test_weighted_aggregation_and_no_alias(self):
        import torch
        from aggregation import weighted_average
        a = {'w': torch.tensor([2.]), 'count': torch.tensor(2)}
        b = {'w': torch.tensor([6.]), 'count': torch.tensor(4)}
        result = weighted_average([(1, a), (3, b)])
        self.assertEqual(result['w'].item(), 5.)
        self.assertEqual(result['count'].dtype, torch.int64)
        result['w'].zero_()
        self.assertEqual(a['w'].item(), 2.)
        with self.assertRaises(ValueError):
            weighted_average([(0, a)])

    def test_pacfl_operates_on_messages(self):
        import numpy as np
        from clustering import PACFL
        groups = PACFL(20)({10: np.array([[1.], [0.]]),
                            20: np.array([[1.], [0.]]),
                            30: np.array([[0.], [1.]])})
        self.assertEqual(groups[10], groups[20])
        self.assertNotEqual(groups[10], groups[30])

    def test_server_has_no_client_or_evaluation_dependency(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path('server.py').read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse({'clients', 'test_loader', 'evaluate', 'update'} & names)


if __name__ == '__main__':
    unittest.main()
