import json
import unittest

import numpy as np

from secfl.leakage import DropoutPolicy, LeakageLog


class LeakageTests(unittest.TestCase):
    def test_abort_policy(self):
        p = DropoutPolicy('abort', threshold=2)
        self.assertTrue(p.accept({'a', 'b', 'c'}))
        self.assertTrue(p.accept({'c', 'b', 'a'}))
        self.assertFalse(p.accept({'a', 'b'}))             # changed set -> discard round
        self.assertFalse(p.accept({'a'}))                  # below t

    def test_dp_noise_variance(self):
        p = DropoutPolicy('dp', threshold=4, sigma=2.0)
        rng = np.random.default_rng(0)
        total = sum(p.client_noise(200_000, rng) for _ in range(4))    # exactly t survivors
        self.assertAlmostEqual(total.std(), 2.0, delta=.02)
        self.assertTrue(p.accept({'a', 'b', 'c', 'd'}))
        self.assertFalse(np.any(DropoutPolicy('leak', 2).client_noise(5)))

    def test_leak_log(self):
        log = LeakageLog()
        log.record(1, ['cat'], {'cat': 10}, {'cat': 2}, {'cat': np.ones(4)}, {'a', 'b', 'c'}, True)
        row = log.record(2, ['cat'], {'cat': 7}, {'cat': 2}, {'cat': np.ones(4)}, {'a', 'c'}, True,
                         previous_survivors={'a', 'b', 'c'})
        self.assertEqual(row['dropped_since_last'], ['b'])
        self.assertEqual(row['aggregate_grad_norm']['cat'], 2.0)
        self.assertEqual(len(json.loads(log.to_json())), 2)

    def test_validation(self):
        for args in (('x', 2), ('dp', 2), ('abort', 0)):
            with self.assertRaises(ValueError):
                DropoutPolicy(*args)


if __name__ == '__main__':
    unittest.main()
