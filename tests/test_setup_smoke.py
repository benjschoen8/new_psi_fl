"""Setup benchmark must stay independent of the training entry point."""
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import setup_smoke


class SetupSmokeTests(unittest.TestCase):
    def test_default_client_sweep_saves_completed_trials_before_failure(self):
        def trial(method, labels, samples, *args, **kwargs):
            n = len(labels)
            if n == 50:
                raise RuntimeError('interrupted last trial')
            return dict(method=method, clients=n, union_size=4, setup_wall_seconds=1.,
                        estimated_upload_bytes_per_client=10, estimated_download_bytes_per_client=20,
                        backend='ideal-functionality')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            with patch('setup_smoke.benchmark', side_effect=trial), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'interrupted last trial'):
                    setup_smoke.main(['--data', 'synthetic', '--simulate', '--methods', 'exact', '--out', directory])
            report = json.loads((Path(directory) / 'setup.json').read_text())
            self.assertEqual(report['config']['clients'], [7, 10, 30, 50])
            self.assertEqual([r['clients'] for r in report['results']], [7, 10, 30])
            self.assertEqual(len((Path(directory) / 'setup.csv').read_text().splitlines()), 4)

    def test_certificates_cover_50_parties_and_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            player = root / 'Player-Data'
            player.mkdir()
            for i in range(16):
                for suffix in ('pem', 'key'):
                    (player / f'P{i}.{suffix}').touch()
            with patch('setup_smoke.subprocess.run') as run:
                setup_smoke.ensure_certificates(root, 50)
                self.assertEqual(run.call_args.args[0], ['bash', 'Scripts/setup-ssl.sh', '50'])
            for i in range(16, 50):
                for suffix in ('pem', 'key'):
                    (player / f'P{i}.{suffix}').touch()
            with patch('setup_smoke.subprocess.run') as run:
                setup_smoke.ensure_certificates(root, 50)
                run.assert_not_called()

    def test_reuses_valid_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'compile.py').touch()
            (root / 'shamir-party.x').touch(mode=0o755)
            with patch.dict(os.environ, {'MPSPDZ': directory}), patch('setup_smoke.subprocess.run') as run:
                self.assertEqual(setup_smoke.ensure_mpspdz(), root.resolve())
                run.assert_not_called()

    def test_missing_installation_downloads_and_sets_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def install(*args, **kwargs):
                (root / 'compile.py').touch()
                (root / 'shamir-party.x').touch(mode=0o755)
                return subprocess.CompletedProcess(args, 0, str(root) + '\n')
            with patch.dict(os.environ, {'MPSPDZ': str(root / 'missing')}):
                with patch('setup_smoke.platform.system', return_value='Linux'), patch('setup_smoke.platform.machine', return_value='x86_64'), patch('setup_smoke.subprocess.run', side_effect=install) as run:
                    self.assertEqual(setup_smoke.ensure_mpspdz(), root.resolve())
                    self.assertEqual(os.environ['MPSPDZ'], str(root.resolve()))
                    self.assertEqual(Path(run.call_args.args[0][1]).name, 'get_mpspdz.sh')

    def test_install_failure_does_not_fall_back(self):
        with patch.dict(os.environ, {'MPSPDZ': '/missing/mp-spdz'}):
            with patch('setup_smoke.platform.system', return_value='Linux'), patch('setup_smoke.platform.machine', return_value='x86_64'), patch('setup_smoke.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'installer')):
                with self.assertRaisesRegex(RuntimeError, 'MP-SPDZ'):
                    setup_smoke.ensure_mpspdz()

    def test_gx10_selects_source_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'compile.py').touch()
            (root / 'shamir-party.x').touch(mode=0o755)
            with patch.dict(os.environ, {'MPSPDZ': ''}), patch('setup_smoke.platform.system', return_value='Linux'), patch('setup_smoke.platform.machine', return_value='aarch64'), patch('setup_smoke.build_mpspdz_from_source', return_value=root) as build, patch('setup_smoke.subprocess.run') as binary:
                self.assertEqual(setup_smoke.ensure_mpspdz(), root)
                build.assert_called_once()
                binary.assert_not_called()

    def test_cached_source_build_initializes_ssl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'compile.py').touch()
            (root / 'shamir-party.x').touch(mode=0o755)
            with patch.dict(os.environ, {'MPSPDZ_BUILD_DIR': directory}), patch('setup_smoke.subprocess.run') as run:
                self.assertEqual(setup_smoke.build_mpspdz_from_source(), root)
                self.assertEqual(run.call_args.args[0], ['bash', 'Scripts/setup-ssl.sh', '16'])

    def test_setup_only_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            # Reject even an import of training, including indirect imports.
            code = """
import runpy, sys
class BlockTraining:
    def find_spec(self, fullname, *args):
        if fullname in {'torch', 'training', 'client', 'secure_cbn', 'secure_code_no_cluster'}:
            raise AssertionError('training dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
sys.argv = ['setup_smoke', '--clients', '3', '--labels', '2', '--bucket-bits', '10',
            '--data', 'synthetic', '--simulate', '--out', sys.argv[1]]
runpy.run_module('setup_smoke', run_name='__main__')
"""
            env = dict(os.environ)
            env.pop('MPSPDZ', None)
            result = subprocess.run([sys.executable, '-c', code, directory], env=env,
                                    cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads((Path(directory) / 'setup.json').read_text())['results']
            self.assertEqual([r['method'] for r in rows], ['plain', 'exact'])
            self.assertEqual(rows[0]['union_size'], rows[1]['union_size'])
            self.assertGreater(rows[1]['setup_wall_seconds'], 0)
            self.assertGreater(rows[1]['estimated_upload_bytes_per_client'], 0)
            self.assertIsNone(rows[1]['mpc_measured'])
            self.assertEqual(rows[1]['backend'], 'ideal-functionality')
            self.assertTrue((Path(directory) / 'setup.csv').exists())

    def test_real_inputs_reuse_partition_metadata_and_sampling(self):
        import setup
        labels_by_dataset = {'MNIST': ('0', '1'), 'EMNIST': ('A', 'a'), 'CIFAR10': ('cat', 'dog')}
        received = []

        def load(args, root, **config):
            received.append(args)
            self.assertEqual(root, '/existing/data')
            self.assertEqual(config['dirichlet_alpha'], .1)
            return ({name: [{'train': SimpleNamespace(dataset=SimpleNamespace(classes=names))}
                            for _ in range(getattr(args, 'num_train_' + name.lower()))]
                     for name, names in labels_by_dataset.items()}, {}, {})

        # Stub only disk/tensor boundaries; real benchmark orchestration and original
        # label_names resolve the client-local label order.
        fake_fl = SimpleNamespace(load_partitioned_datasets=load)
        fake_conf = SimpleNamespace(OmegaConf=SimpleNamespace(
            load=lambda p: {'dirichlet_alpha': .1}, to_container=lambda c, **kw: c))
        args = SimpleNamespace(seed=2026, data_root=Path('/existing/data'), exp_conf=Path('config.yaml'),
                               class_subsets=None, class_share='split', noniid_partition='dirichlet',
                               samples_per_label=16, fuzzy_langs='en0,en1', datasets='MNIST,EMNIST,CIFAR10')
        with patch.dict(sys.modules, {'fl_datasets': fake_fl, 'omegaconf': fake_conf}):
            with patch.object(setup, 'seed_all'), patch.object(setup, 'label_samples',
                    side_effect=lambda loader, names, k: {names[0]: setup_smoke.np.zeros((k, 3, 32, 32))}) as sample:
                names, samples, keywords, meta = setup_smoke.real_inputs(5, args)
        self.assertEqual(meta['dataset_clients'], {'MNIST': 1, 'EMNIST': 3, 'CIFAR10': 1})   # by label slots needed
        self.assertEqual(received[0].num_new_clients, 0)
        self.assertEqual(received[0].num_train_usps, 0)
        self.assertEqual(names[-1], ['cat', 'dog'])
        self.assertEqual(names[2], ['A', 'a'])
        self.assertEqual(keywords[0]['0'], 'zero')
        from rt_descriptions import keyword
        self.assertEqual(keywords[1]['A'], keyword('EMNIST', 'A', 'en1'))
        self.assertEqual(sample.call_count, 5)
        self.assertEqual(len(samples[0]['0']), 16)

    def test_default_cli_selects_real_components_and_production_buckets(self):
        row = dict(method='exact', clients=3, union_size=2, setup_wall_seconds=1.,
                   estimated_upload_bytes_per_client=10, estimated_download_bytes_per_client=20,
                   backend='mp-spdz')
        prepared = ([['cat']] * 3, [{'cat': setup_smoke.np.zeros((1, 3, 32, 32))}] * 3,
                    [{'cat': 'Cat'}] * 3, {'source': 'real', 'data_load_partition_seconds': 2., 'sampling_seconds': .5})
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch('setup_smoke.real_inputs', return_value=prepared) as inputs, patch('setup_smoke.fixture') as synthetic, patch('setup_smoke.benchmark', return_value=row) as bench, patch('setup_smoke.ensure_mpspdz', return_value=Path('/mp-spdz')), patch('setup_smoke.ensure_certificates'):
                setup_smoke.main(['--clients', '3', '--datasets', 'MNIST,EMNIST,CIFAR10', '--num-train-cifar10stl10', '0',
                                 '--methods', 'exact', '--out', directory])
                inputs.assert_called_once()
                synthetic.assert_not_called()
                self.assertEqual(bench.call_args.args[3], 20)
                self.assertEqual(bench.call_args.kwargs['keywords'], prepared[2])
            report = json.loads((Path(directory) / 'setup.json').read_text())
            self.assertEqual(report['results'][0]['data_source'], 'real')
            self.assertEqual(report['results'][0]['sampling_seconds'], .5)

    def test_shared_sampler_preserves_local_order_caps_and_missing_labels(self):
        from setup import label_samples
        np = setup_smoke.np
        # Tensor-like batches isolate the sampler from PyTorch installation and disk I/O.
        batch = [SimpleNamespace(numpy=lambda i=i: np.full((3, 2, 2), i)) for i in range(5)]
        data = SimpleNamespace(DataLoader=lambda dataset, **kwargs: [(batch, np.array([1, 0, 1, 1, 0]))])
        with patch.dict(sys.modules, {'torch.utils.data': data}):
            sampled = label_samples(SimpleNamespace(dataset=object()), ['dog', 'cat', 'absent'], 2)
        self.assertEqual(set(sampled), {'dog', 'cat'})
        np.testing.assert_array_equal(sampled['cat'][:, 0, 0, 0], [0, 2])
        np.testing.assert_array_equal(sampled['dog'][:, 0, 0, 0], [1, 4])


if __name__ == '__main__':
    unittest.main()
