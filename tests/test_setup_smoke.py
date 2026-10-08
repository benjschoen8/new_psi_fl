"""Setup benchmark must stay independent of the training entry point."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import setup_smoke


class SetupSmokeTests(unittest.TestCase):
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
                with patch('setup_smoke.subprocess.run', side_effect=install) as run:
                    self.assertEqual(setup_smoke.ensure_mpspdz(), root.resolve())
                    self.assertEqual(os.environ['MPSPDZ'], str(root.resolve()))
                    self.assertEqual(Path(run.call_args.args[0][1]).name, 'get_mpspdz.sh')

    def test_install_failure_does_not_fall_back(self):
        with patch.dict(os.environ, {'MPSPDZ': '/missing/mp-spdz'}):
            with patch('setup_smoke.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'installer')):
                with self.assertRaisesRegex(RuntimeError, 'MP-SPDZ'):
                    setup_smoke.ensure_mpspdz()

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
            '--simulate', '--out', sys.argv[1]]
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


if __name__ == '__main__':
    unittest.main()
