import unittest

import numpy as np
import torch

from secure_main import build_synthetic
from secure_code_no_cluster import run, TinyCodeGenerator, TinyLocalDiscriminator, seeded
from smoke import TinyClassifier

CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4)


class SecureCodeMainTests(unittest.TestCase):
    def go(self, agg):
        clients, spaces, tests = build_synthetic(CONFIG)
        return run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator, TinyClassifier,
                   CONFIG, ['bird', 'cat', 'dog', 'horse', 'ship', 'truck'], agg=agg, rounds=2, code_dim=8, union='secagg')

    def run_with(self, **kw):
        torch.manual_seed(7); np.random.seed(7)                            # data + run both seeded
        clients, spaces, tests = build_synthetic(CONFIG)
        return run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator, TinyClassifier,
                   CONFIG, ['bird', 'cat', 'dog', 'horse', 'ship', 'truck'], agg='plain', code_dim=8, seed=3, **dict(dict(union='secagg'), **kw))

    def test_resume_is_identical_to_uninterrupted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            full = self.run_with(rounds=3)
            self.run_with(rounds=2, checkpoint_dir=d)
            resumed = self.run_with(rounds=3, resume=f'{d}/checkpoint_last.pt')
        self.assertEqual(resumed['setup']['resumed_from'], 2)
        np.testing.assert_array_equal(resumed['theta'], full['theta'])
        self.assertEqual([h['accuracy'] for h in resumed['history']], [h['accuracy'] for h in full['history']])

    def test_parallel_equals_sequential(self):
        a = self.run_with(rounds=2, workers=1)
        b = self.run_with(rounds=2, workers=3)
        np.testing.assert_array_equal(a['theta'], b['theta'])

    def test_parallel_workers_and_keep_all(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            out = self.run_with(rounds=2, workers=3, checkpoint_dir=d, keep_all=True)
            self.assertEqual(sorted(os.listdir(d)), ['checkpoint_last.pt', 'round_0001.pt', 'round_0002.pt', 'setup.json'])
        self.assertEqual(len(out['history']), 2)

    def test_resume_rejects_other_experiment(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.run_with(rounds=1, checkpoint_dir=d)
            clients, spaces, tests = build_synthetic(CONFIG)
            with self.assertRaises(ValueError):
                run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator, TinyClassifier,
                    CONFIG, ['bird', 'cat', 'dog', 'horse', 'ship', 'truck'], agg='secagg', code_dim=8, rounds=2, union='secagg',
                    resume=f'{d}/checkpoint_last.pt')

    def test_public_seed_same_trunk_everywhere(self):
        f = seeded(lambda: TinyCodeGenerator(8, 4))
        a, b = f(), f()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))

    def test_pipelines(self):
        for agg in ('plain', 'secagg'):
            result = self.go(agg)
            setup, last = result['setup'], result['history'][-1]
            self.assertEqual(setup['labels'], 3)                        # cat, dog, ship
            self.assertEqual(setup['union']['holders'], {'cat': 2, 'dog': 2, 'ship': 2})
            self.assertIn('broadcast', result['history'][1]['bytes'])
            self.assertTrue(0 <= last['accuracy'] <= 1)
            if agg == 'secagg':                                          # upload = trunk only, fixed size
                self.assertEqual(last['bytes']['upload'], 3 * (setup['trunk_params'] + 2) * 8)


if __name__ == '__main__':
    unittest.main()
