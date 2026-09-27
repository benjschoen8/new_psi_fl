import tempfile
import unittest

import numpy as np
import torch

from secure_code_cluster import run, build_patterned, encode_upload, decode_sum, TinyCodeGenerator, TinyLocalDiscriminator
from smoke import TinyClassifier

CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4)
DICTIONARY = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']


def go(**kw):
    torch.manual_seed(7); np.random.seed(7)
    clients, spaces, tests = build_patterned(CONFIG)
    kw = dict(dict(cluster='plain', agg='plain', rounds=2, code_dim=8, basis_budget=2, seed=3, union='secagg'), **kw)
    return run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator, TinyClassifier,
               CONFIG, DICTIONARY, **kw)


class SecureCodeClusterTests(unittest.TestCase):
    def test_upload_sum_is_group_mean_without_counts(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((3, 5))
        vecs = [encode_upload(u[0], 0, 2, 1e3, 1 / 2), encode_upload(u[1], 0, 2, 1e3, 1 / 2),
                encode_upload(u[2], 1, 2, 1e3, 1.0)]
        with np.errstate(over='ignore'):
            total = np.sum(vecs, axis=0, dtype=np.uint64)
        np.testing.assert_allclose(decode_sum(total, 2), [u[:2].mean(0), u[2]], atol=1e-6)

    def test_private_equals_plain_grouping_and_training(self):
        private, plain = go(cluster='private', workers=2), go(cluster='plain')
        self.assertEqual(private['grouping']['clients'], plain['grouping']['clients'])
        np.testing.assert_array_equal(private['grouping']['labels'], plain['grouping']['labels'])
        self.assertEqual(private['setup']['groups'], 2)
        self.assertEqual(private['setup']['aggregator_view'], {0: ['cat', 'dog'], 1: ['ship', 'truck']})
        np.testing.assert_array_equal(private['thetas'], plain['thetas'])
        self.assertIn('mpc_and_gates', private['setup']['grouping_stats'])

    def test_secagg_matches_plain_aggregation(self):
        a, b = go(agg='plain'), go(agg='secagg')
        np.testing.assert_allclose(a['thetas'], b['thetas'], atol=1e-6)
        G, d = b['thetas'].shape
        self.assertEqual(b['history'][-1]['bytes']['upload'], 4 * G * d * 8)     # fixed size, no counts

    def test_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = go(rounds=3)
            go(rounds=2, checkpoint_dir=tmp)
            resumed = go(rounds=3, resume=f'{tmp}/checkpoint_last.pt')
        self.assertEqual(resumed['setup']['resumed_from'], 2)
        np.testing.assert_array_equal(resumed['thetas'], full['thetas'])
        self.assertEqual([h['accuracy'] for h in resumed['history']], [h['accuracy'] for h in full['history']])

    def test_original_pacfl_option(self):
        self.assertEqual(go(cluster='original', rounds=1)['setup']['groups'], 2)


if __name__ == '__main__':
    unittest.main()
