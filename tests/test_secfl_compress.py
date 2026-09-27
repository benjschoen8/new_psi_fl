import unittest

import numpy as np
import torch

from secfl import compress
from secfl.label_gan import pair_state, load_pair_state
from secfl.secagg import run_secagg


class CompressTests(unittest.TestCase):
    def test_stochastic_rounding_is_unbiased(self):
        rng = np.random.default_rng(0)
        u = np.array([.013, -.021, .0004, .05, -.05])
        mean = np.mean([compress.quantize(u, .05, rng) for _ in range(20000)], 0) / compress.LEVELS * .05
        np.testing.assert_allclose(mean, u, atol=3e-4)
        self.assertEqual(compress.quantize(np.array([9., -9.]), .05, rng).tolist(), [127, -127])   # clipped

    def test_secagg_16bit_mean_per_slot(self):
        rng = np.random.default_rng(1)
        M, k, scales = 3, 50, {0: .1, 1: .1, 2: .1}
        ups = {0: {0: rng.uniform(-.1, .1, k), 2: rng.uniform(-.1, .1, k)},
               1: {0: rng.uniform(-.1, .1, k)},
               2: {}}                                                    # holds nothing: all zeros
        vectors = {c: compress.encode(u, M, k, scales, rng) for c, u in ups.items()}
        total, _ = run_secagg(vectors, threshold=2, modulus_bits=compress.BITS, session=b't')
        means, holders = compress.decode(total, M, k, scales)
        self.assertEqual(holders, {0: 2, 1: 0, 2: 1})
        self.assertEqual(set(means), {0, 2})
        np.testing.assert_allclose(means[0], (ups[0][0] + ups[1][0]) / 2, atol=.1 / 127)
        np.testing.assert_allclose(means[2], ups[0][2], atol=.1 / 127)

    def test_no_overflow_up_to_258_clients(self):
        rng = np.random.default_rng(2)
        vecs = [compress.encode({0: np.full(4, -1.)}, 1, 4, {0: 1.}, rng) for _ in range(258)]
        with np.errstate(over='ignore'):
            total = np.sum(vecs, axis=0, dtype=np.uint64)
        means, holders = compress.decode(total, 1, 4, {0: 1.})
        self.assertEqual(holders[0], 258)
        np.testing.assert_allclose(means[0], -1.)

    def test_keep_index_is_public_and_changes_per_round(self):
        a, b = compress.keep_index(1000, .1, 3), compress.keep_index(1000, .1, 3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.size, 100)
        self.assertFalse(np.array_equal(a, compress.keep_index(1000, .1, 4)))
        np.testing.assert_array_equal(compress.keep_index(7, 1., 0), np.arange(7))

    def test_scales_are_per_tensor(self):
        spec = [('a', (4,), np.float32), ('b', (2,), np.float32)]
        init = np.array([0., 0., 0., 0., 2., 2.])
        np.testing.assert_allclose(compress.initial_scales(spec, init, .05), [.05] * 4 + [1., 1.])
        delta = np.array([.01, 0, .01, 0, .3, 0])
        sc = compress.next_scales(spec, delta, kept=np.array([0, 2, 4]))
        np.testing.assert_allclose(sc, [.04] * 4 + [1.2] * 2)             # 4 x RMS over kept coords

    def test_d_stays_local(self):
        g, d = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        state = pair_state(g, d, include_d=False)
        self.assertTrue(all(k.startswith('G.') for k in state))
        before = {k: v.clone() for k, v in d.state_dict().items()}
        load_pair_state(torch.nn.Linear(2, 2), d, state)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in d.state_dict().items()))


class PipelineSizeTests(unittest.TestCase):
    def test_upload_bytes_and_quality(self):
        from secure_main import run, build_synthetic
        from smoke import TinyGenerator, TinyDiscriminator, TinyClassifier
        config = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4)
        clients, spaces, tests = build_synthetic(config)
        res = run(clients, spaces, tests, lambda: TinyGenerator(1), TinyDiscriminator, TinyClassifier, config,
                  rounds=2, keep_frac=.5)
        G = sum(p.numel() for p in TinyGenerator(1).parameters())
        k = round(G * .5)
        self.assertEqual(res['history'][-1]['bytes']['upload'], 3 * (3 * k + 3) * 2)   # 3 clients, M = 3
        self.assertEqual(res['history'][-1]['active_slots'], 3)


if __name__ == '__main__':
    unittest.main()
