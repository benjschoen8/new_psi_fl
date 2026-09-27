import unittest

import numpy as np

from secfl.settle import settle, flatten, unflatten


class SettleTests(unittest.TestCase):
    def test_threshold_and_weighted_mean(self):
        thetas = {'a': np.ones(3), 'b': np.zeros(2), 'c': np.ones(1)}
        grads = {'a': np.array([10., 20., 30.]), 'b': np.array([4., 4.]), 'c': np.array([0.])}
        N = {'a': 10, 'b': 4, 'c': 0}
        G = {'a': 3, 'b': 1, 'c': 2}
        new, updated = settle(thetas, grads, N, G, lr=.5, threshold=2)
        self.assertEqual(updated, ['a'])
        np.testing.assert_allclose(new['a'], [0.5, 0., -0.5])       # 1 - .5 * [1,2,3]
        np.testing.assert_array_equal(new['b'], thetas['b'])         # below t: unchanged
        np.testing.assert_array_equal(new['c'], thetas['c'])         # N=0: unchanged

    def test_fedavg_of_generator_weights(self):
        glob = np.zeros(3)
        local = {'c1': (np.array([1., 2., 3.]), 10), 'c2': (np.array([3., 2., 1.]), 30)}
        delta_sum = sum(n * -(w - glob) for w, n in local.values())
        new, _ = settle({'L': glob}, {'L': delta_sum}, {'L': 40}, {'L': 2}, lr=1, threshold=2)
        np.testing.assert_allclose(new['L'], (10 * local['c1'][0] + 30 * local['c2'][0]) / 40)

    def test_flatten_roundtrip(self):
        state = {'w': np.arange(6, dtype=np.float32).reshape(2, 3), 'b': np.array([1., 2.])}
        flat, spec = flatten(state)
        back = unflatten(flat, spec)
        self.assertEqual(back['w'].dtype, np.float32)
        np.testing.assert_array_equal(back['w'], state['w'])
        with self.assertRaises(ValueError):
            unflatten(flat[:-1], spec)

    def test_validation(self):
        with self.assertRaises(ValueError):
            settle({'a': np.ones(2)}, {'a': np.ones(3)}, {'a': 1}, {'a': 5}, .1, 1)
        with self.assertRaises(ValueError):
            settle({}, {}, {}, {}, 0, 1)


if __name__ == '__main__':
    unittest.main()
