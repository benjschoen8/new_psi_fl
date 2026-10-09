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

import setup_smoke_hybrid


class SetupSmokeTests(unittest.TestCase):
    def test_pairwise_benchmark_routes_options_and_keeps_unknown_bytes(self):
        measured = dict(compile_seconds=.2, wall_seconds=1., global_MB=.4,
                        matching_seconds=.6, group_seconds=.4, pair_batches=2,
                        bridge_seconds=.1, pair_compile_seconds=.05,
                        group_compile_seconds=.15, pair_sessions=3, max_parallel_pairs=2)
        stats = dict(mpc={'measured': measured}, setup_upload_bytes_per_client=None,
                     setup_download_bytes_per_client=None)
        with patch('setup_smoke_hybrid.client_sets', return_value=[1, 2]), patch(
                'setup_smoke_hybrid.circuit_union_with_keys', return_value=(None, None, None, 1, stats)) as union:
            result = setup_smoke_hybrid.benchmark('exact', [['cat']] * 3, [{}] * 3,
                mpc_mode='pairwise', pair_concurrency=2, pair_workers=4, mpc_timeout=90)
        self.assertEqual(union.call_args.kwargs['mpc_backend'], 'pairwise')
        self.assertEqual(union.call_args.kwargs['mpc_options'],
                         {'pair_concurrency': 2, 'pair_workers': 4, 'timeout': 90,
                          'prefix': 'parallel', 'block_rows': 64, 'group_edabit': False,
                          'group_version': 1, 'group_protocol': 'shamir', 'pair_protocol': 'semi', 'pad_max': 'plain'})
        self.assertEqual(result['backend'], 'mp-spdz-pairwise')
        self.assertEqual(result['mpc_mode'], 'pairwise')
        self.assertEqual(result['pair_sessions'], 3)
        self.assertEqual(result['matching_seconds'], .6)
        self.assertEqual(result['bridge_seconds'], .1)
        self.assertEqual(result['matching_compile_seconds'], .05)
        self.assertEqual(result['group_compile_seconds'], .15)
        self.assertEqual(result['group_prefix'], 'parallel')
        self.assertIsNone(result['estimated_upload_bytes_total'])
        self.assertIsNone(result['estimated_download_bytes_per_client'])

    def test_global_benchmark_keeps_original_options(self):
        stats = dict(mpc={}, setup_upload_bytes_per_client=10,
                     setup_download_bytes_per_client=20)
        with patch('setup_smoke_hybrid.client_sets', return_value=[1, 2]), patch(
                'setup_smoke_hybrid.circuit_union_with_keys', return_value=(None, None, None, 1, stats)) as union:
            result = setup_smoke_hybrid.benchmark('exact', [['cat']] * 3, [{}] * 3,
                                          mpc_mode='global', group_prefix='serial', mpc_timeout=90)
        self.assertEqual(union.call_args.kwargs['mpc_backend'], 'global')
        self.assertEqual(union.call_args.kwargs['mpc_options'], {})
        self.assertEqual(result['group_prefix'], 'serial')

    def test_pairwise_missing_measurements_does_not_fall_back(self):
        stats = dict(mpc={}, setup_upload_bytes_per_client=10,
                     setup_download_bytes_per_client=20)
        with patch('setup_smoke_hybrid.client_sets', return_value=[1, 2]), patch(
                'setup_smoke_hybrid.circuit_union_with_keys', return_value=(None, None, None, 1, stats)):
            with self.assertRaisesRegex(RuntimeError, 'real MP-SPDZ measurements'):
                setup_smoke_hybrid.benchmark('exact', [['cat']] * 3, [{}] * 3, mpc_mode='pairwise')

    def test_both_modes_run_plain_once_then_each_secure_backend(self):
        def trial(method, labels, samples, *args, **kwargs):
            return dict(method=method, clients=len(labels), union_size=2, setup_wall_seconds=1.,
                        estimated_upload_bytes_per_client=None, estimated_download_bytes_per_client=None,
                        backend='mp-spdz-pairwise', mpc_mode=kwargs['mpc_mode'])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch('setup_smoke_hybrid.benchmark', side_effect=trial) as bench, patch(
                    'setup_smoke_hybrid.ensure_mpspdz', return_value=Path('/mp-spdz')), patch(
                    'setup_smoke_hybrid.ensure_certificates'), patch('setup_smoke_hybrid.ensure_pairwise_backend') as backend:
                setup_smoke_hybrid.main(['--data', 'synthetic', '--clients', '3', '--mpc-mode', 'both', '--methods', 'plain', 'exact',
                    '--pair-concurrency', '3', '--pair-workers', '2', '--mpc-timeout', '120',
                    '--group-prefix', 'serial', '--out', directory])
            report = json.loads((Path(directory) / 'setup.json').read_text())
            self.assertIsNone(report['results'][-1]['estimated_upload_bytes_per_client'])
            self.assertNotIn('None', (Path(directory) / 'setup.csv').read_text())
        self.assertEqual([(call.args[0], call.kwargs['mpc_mode']) for call in bench.call_args_list],
                         [('plain', 'global'), ('exact', 'global'), ('exact', 'pairwise')])
        self.assertEqual(bench.call_args.kwargs['pair_concurrency'], 3)
        self.assertEqual(bench.call_args.kwargs['pair_workers'], 2)
        self.assertEqual(bench.call_args.kwargs['mpc_timeout'], 120)
        self.assertTrue(all(call.kwargs['group_prefix'] == 'serial' for call in bench.call_args_list))
        backend.assert_called_once_with(Path('/mp-spdz'), ('semi-party.x', 'shamir-party.x'))

    def test_pairwise_rejects_simulation_and_invalid_limits_before_installation(self):
        for flags in (['--mpc-mode', 'pairwise', '--simulate'], ['--mpc-mode', 'both', '--methods', 'plain', 'exact', '--simulate'],
                      ['--pair-concurrency', '0'], ['--pair-workers', '-1'], ['--mpc-timeout', '0'],
                      ['--mpc-timeout', 'nan'], ['--group-prefix', 'invalid'], ['--group-block-rows', '0']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), patch(
                    'setup_smoke_hybrid.ensure_mpspdz') as install:
                with self.assertRaises(SystemExit):
                    setup_smoke_hybrid.main(['--data', 'synthetic', '--clients', '3', *flags])
                install.assert_not_called()

    def test_failed_trial_saves_diagnostic_and_preserves_completed_results(self):
        for first_fails in [True, False]:
            good = dict(method='exact', clients=3, union_size=2, setup_wall_seconds=1.,
                        estimated_upload_bytes_per_client=None, estimated_download_bytes_per_client=None,
                        backend='mp-spdz-pairwise', mpc_mode='pairwise')
            trials = [setup_smoke_hybrid.MPCSessionError('stage=graph-Log party=0 signal=SIGKILL')]
            if not first_fails:
                trials.insert(0, good)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                with patch('setup_smoke_hybrid.benchmark', side_effect=trials), patch(
                        'setup_smoke_hybrid.ensure_mpspdz', return_value=Path('/mp-spdz')), patch(
                        'setup_smoke_hybrid.ensure_certificates'), patch('setup_smoke_hybrid.ensure_pairwise_backend'):
                    with self.assertRaisesRegex(RuntimeError, 'SIGKILL'):
                        setup_smoke_hybrid.main(['--data', 'synthetic', '--clients', '3', '--methods', 'exact',
                                                '--repeats', '2', '--out', directory])
                report = json.loads((Path(directory) / 'setup.json').read_text())
                self.assertEqual(len(report['results']), 0 if first_fails else 1)
                self.assertIn('SIGKILL', report['failures'][0]['error'])

    def test_pairwise_backend_reuses_binary_or_builds_native_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Makefile').touch()
            def build(*args, **kwargs):
                (root / 'semi-party.x').touch(mode=0o755)
            with patch.dict(os.environ, {'MPSPDZ_JOBS': '2'}), patch(
                    'setup_smoke_hybrid.subprocess.run', side_effect=build) as run:
                setup_smoke_hybrid.ensure_pairwise_backend(root)
                self.assertEqual(run.call_args.args[0], ['make', '-j', '2', 'semi-party.x'])
                self.assertEqual(run.call_args.kwargs['cwd'], root)
                run.reset_mock()
                setup_smoke_hybrid.ensure_pairwise_backend(root)
                run.assert_not_called()

    def test_pairwise_build_failure_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Makefile').touch()
            with patch('setup_smoke_hybrid.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'make')):
                with self.assertRaisesRegex(RuntimeError, 'semi-party'):
                    setup_smoke_hybrid.ensure_pairwise_backend(root)


class GarbledOptionTests(unittest.TestCase):
    def test_garbled_pair_versions_route_options(self):
        calls = []

        def fake(*args, **kwargs):
            calls.append(kwargs)
            raise RuntimeError('stop')
        with patch('setup_smoke_hybrid.circuit_union_with_keys', side_effect=fake), \
                patch('setup_smoke_hybrid.client_sets', return_value={}), self.assertRaises(RuntimeError):
            setup_smoke_hybrid.benchmark('fuzzy', [['a']] * 3, [{}] * 3, keywords=[{'a': 'a'}] * 3,
                                         pair_protocol='simhash', group_version=2, pca_dim=64, simhash_bits=128)
        options = calls[0]['mpc_options']
        self.assertEqual((options['pair_protocol'], options['simhash_bits'], options['pca_dim']), ('simhash', 128, 64))


class SetupCommTests(unittest.TestCase):
    def test_model_adds_mpc_and_union_traffic(self):
        row = dict(clients=2, mpc_measured=dict(client_sent_MB=[1., 3.], client_received_MB=[3., 1.],
                                                client_rounds=[10, 20]),
                   secagg_accounted_bytes=dict(payload_up=2e6, control_up=0, control_down=2e6))
        out = setup_smoke_hybrid.setup_comm(row, mbps=8., rtt_ms=100.)
        self.assertAlmostEqual(out['comm_MB_per_client_mean'], 6.)           # 4 MPC + 1 up + 1 down
        self.assertAlmostEqual(out['comm_seconds_model_max'], 6. + 2.3)      # 6 MB at 1 MB/s, 23 rounds


class UnionQualityTests(unittest.TestCase):
    def test_pair_mcc(self):
        labels = [['0', 'cat'], ['0', 'dog'], ['cat']]
        good = [{'0': 0, 'cat': 1}, {'0': 0, 'dog': 2}, {'cat': 1}]
        bad = [{'0': 0, 'cat': 1}, {'0': 0, 'dog': 1}, {'cat': 1}]          # dog merged with cat
        self.assertEqual(setup_smoke_hybrid.union_quality(labels, good, 3)['pair_mcc'], 1.)
        q = setup_smoke_hybrid.union_quality(labels, bad, 2)
        self.assertLess(q['pair_mcc'], 1.)
        self.assertEqual((q['pair_fp'], q['union_exact']), (2, False))


class MpcModelTests(unittest.TestCase):
    def test_model_matches_fit_points_and_runs_without_mpspdz(self):
        from label_union import mpc_model
        self.assertLess(abs(mpc_model.he(62, 64)[0] - 73.2) / 73.2, .05)
        self.assertLess(abs(mpc_model.group(10, 62, 1)[0] * 10 - 495) / 495, .1)
        est = mpc_model.estimate(10, 62, 48, 45, 1)
        self.assertAlmostEqual(sum(est['client_sent_MB']), est['global_MB'], delta=1e-6 * est['global_MB'])
        with tempfile.TemporaryDirectory() as out, patch.dict(os.environ, {}, clear=False):
            os.environ.pop('MPSPDZ', None)
            with contextlib.redirect_stdout(io.StringIO()):
                setup_smoke_hybrid.main(['--data', 'synthetic', '--clients', '3', '--labels', '4', '--methods', 'exact',
                                         '--pair-protocol', 'hegc', '--group-protocol', 'atlas', '--mpc-model',
                                         '--out', out])
            row = json.loads((Path(out) / 'setup.json').read_text())['results'][0]
        self.assertEqual(row['backend'], 'mpc-model')
        self.assertGreater(row['setup_total_seconds_model'], row['comm_seconds_model_max'])
        self.assertEqual(row['pair_mcc'], 1.)
