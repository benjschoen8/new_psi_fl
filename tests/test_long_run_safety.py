"""Checks that matter before an hours-long run: union MCC, ground-truth accuracy, resume after a
crash, log hygiene, and that the plots/CSVs show exactly what metrics.jsonl holds."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluation import evaluate_global
from secure_main import build_synthetic
from secure_code_no_cluster import (run, union_relations, trim_log, TinyCodeGenerator, TinyLocalDiscriminator)
import secure_code_cluster
from smoke import TinyClassifier

CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4)
DICTIONARY = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']
ROOT = Path(__file__).resolve().parents[1]


def no_cluster(**kw):
    torch.manual_seed(7); np.random.seed(7)
    clients, spaces, tests = build_synthetic(CONFIG)
    return run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator, TinyClassifier,
               CONFIG, DICTIONARY, **dict(dict(agg='plain', rounds=2, code_dim=8, seed=3, union='secagg'), **kw)), spaces, tests


def cluster(**kw):
    torch.manual_seed(7); np.random.seed(7)
    clients, spaces, tests = secure_code_cluster.build_patterned(CONFIG)
    kw = dict(dict(cluster='plain', agg='plain', rounds=2, code_dim=8, basis_budget=2, seed=3, union='secagg'), **kw)
    return secure_code_cluster.run(clients, spaces, tests, lambda: TinyCodeGenerator(8, 4), TinyLocalDiscriminator,
                                   TinyClassifier, CONFIG, DICTIONARY, **kw), spaces, tests


def recount(model, spaces, tests, existing):
    """Independent accuracy: predicted class name == true label name, over every test sample."""
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for _, cid, loader in tests:
            for x, y in loader:
                out = model(x)
                pred = (out[1] if isinstance(out, tuple) else out).argmax(1)
                correct += sum(existing[int(p)] == spaces[cid][int(t)] for p, t in zip(pred, y))
                total += len(y)
    return correct / total


class Constant(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.k, self.n = k, n

    def forward(self, x):
        out = torch.zeros(len(x), self.n)
        out[:, self.k] = 1
        return out


def cli(*args):
    return subprocess.run([sys.executable, '-m', *args], cwd=ROOT, capture_output=True, text=True)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


class UnionAndAccuracyTests(unittest.TestCase):
    def test_union_mcc_is_one_with_expected_pair_counts(self):
        result, _, _ = no_cluster()
        um = result['setup']['union_metrics']
        # union {cat, dog, ship} over the 6-entry dictionary; bird, horse, truck held by nobody
        self.assertEqual((um['true_positive'], um['false_positive'], um['false_negative'], um['true_negative']),
                         (3, 0, 0, 3))
        self.assertEqual(um['mcc'], 1.0)
        self.assertTrue(um['exact'] and um['holders_exact'])                 # holder counts 2, 2, 2 too
        self.assertEqual(result['history'][-1]['union_metrics'], um)

    def test_reported_accuracy_equals_independent_recount(self):
        for result, spaces, tests in (no_cluster(rounds=3), cluster(rounds=3)):
            existing = [x for x in DICTIONARY if x in result['setup']['union']['holders']]
            last = result['history'][-1]
            self.assertAlmostEqual(last['accuracy'], recount(result['model'], spaces, tests, existing))
            self.assertAlmostEqual(last['old_acc'], last['accuracy'])          # exact union: both agree
            self.assertEqual(last['evaluation']['samples'], sum(len(l.dataset) for _, _, l in tests))
            self.assertEqual(last['evaluation']['ambiguous_predictions'], 0)

    def test_known_answer_accuracy(self):
        _, spaces, tests = no_cluster(rounds=1)
        existing = ['cat', 'dog', 'ship']
        predicted, truth = union_relations(list(spaces), [spaces[c] for c in spaces], {x: k for k, x in enumerate(existing)})
        ev = evaluate_global(Constant(1, 3), tests, predicted, truth)          # always "dog"
        self.assertAlmostEqual(ev['ground_truth_acc'], 16 / 48)                # dog: 8 images at clients 0 and 1
        self.assertAlmostEqual(ev['by_dataset']['synthetic']['accuracy'], 16 / 48)

    def test_wrong_union_is_caught(self):
        _, spaces, tests = no_cluster(rounds=1)
        ids, names = list(spaces), [spaces[c] for c in spaces]
        predicted, truth = union_relations(ids, names, {'cat': 0, 'dog': 0, 'ship': 1})    # cat and dog merged
        ev = evaluate_global(Constant(0, 2), tests, predicted, truth)
        self.assertEqual(ev['ground_truth_acc'], 0)                            # merged class never counts
        self.assertGreater(ev['old_acc'], 0)                                   # old metric would hide it

    def test_cluster_union_mcc(self):
        result, _, _ = cluster()
        um = result['setup']['union_metrics']
        self.assertEqual((um['true_positive'], um['true_negative'], um['mcc']), (4, 2, 1.0))   # cat dog ship truck
        self.assertTrue(um['holders_exact'])                                   # secagg union: counts are right
        mpc = cluster(union='mpc')[0]['setup']['union_metrics']
        self.assertEqual((mpc['true_positive'], mpc['mcc'], mpc['exact']), (4, 1.0, True))
        self.assertNotIn('holders_exact', mpc)                                 # mpc union: no counts exist


class CrashResumeTests(unittest.TestCase):
    def test_crash_mid_run_then_resume_is_identical(self):
        full, _, _ = no_cluster(rounds=4)

        def crash(row):
            if row['round'] == 3:
                raise RuntimeError('simulated crash after round 3 was trained')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                no_cluster(rounds=4, checkpoint_dir=d, record=crash)
            self.assertTrue((Path(d) / 'setup.json').exists())
            resumed, _, _ = no_cluster(rounds=4, checkpoint_dir=d, resume=f'{d}/checkpoint_last.pt')
        self.assertEqual(resumed['setup']['resumed_from'], 2)
        np.testing.assert_array_equal(resumed['theta'], full['theta'])
        self.assertEqual([h['accuracy'] for h in resumed['history']], [h['accuracy'] for h in full['history']])

    def test_cluster_crash_then_resume_is_identical(self):
        full, _, _ = cluster(rounds=3)

        def crash(row):
            if row['round'] == 2:
                raise RuntimeError('crash')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                cluster(rounds=3, checkpoint_dir=d, record=crash)
            resumed, _, _ = cluster(rounds=3, checkpoint_dir=d, resume=f'{d}/checkpoint_last.pt')
        np.testing.assert_array_equal(resumed['thetas'], full['thetas'])
        self.assertEqual([h['accuracy'] for h in resumed['history']], [h['accuracy'] for h in full['history']])

    def test_trim_log(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'metrics.jsonl'
            p.write_text(''.join(json.dumps(dict(round=r)) + '\n' for r in (1, 2, 3, 4)))
            trim_log(p, 2)
            self.assertEqual([r['round'] for r in rows(p)], [1, 2])
            trim_log(Path(d) / 'missing.jsonl', 2)                             # no file: no error


class CliAndPlotTests(unittest.TestCase):
    def test_no_cluster_cli_resume_log_and_plot(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / 'secagg'
            base = ['secure_code_no_cluster', '--smoke', '--agg', 'secagg', '--no-progress', '--output', str(run_dir)]
            self.assertEqual(cli(*base, '--rounds', '2').returncode, 0)
            refuse = cli(*base, '--rounds', '3')                               # would overwrite a checkpoint
            self.assertNotEqual(refuse.returncode, 0)
            self.assertIn('--resume', refuse.stderr)
            with (run_dir / 'metrics.jsonl').open('a') as f:                   # row of a round lost in a crash
                f.write(json.dumps(dict(round=3, accuracy=-1)) + '\n')
            res = cli(*base, '--rounds', '3', '--resume', str(run_dir / 'checkpoint_last.pt'))
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            log = rows(run_dir / 'metrics.jsonl')
            self.assertEqual([r['round'] for r in log], [1, 2, 3])
            self.assertTrue(all(r['accuracy'] >= 0 for r in log))
            um = json.loads((run_dir / 'evaluator' / 'union_metrics.json').read_text())
            self.assertTrue(um['exact'])
            setup = json.loads((run_dir / 'setup.json').read_text())
            self.assertNotIn('union_metrics', setup)                       # experimenter data: evaluator/ only
            self.assertEqual(setup['resumed_from'], 2)

            p = cli('tests.plot_secure', '--runs', str(run_dir), '--out', d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertTrue((Path(d) / 'secure.png').stat().st_size > 10000)
            with (Path(d) / 'secure.csv').open() as f:
                table = list(csv.DictReader(f))
            self.assertEqual([int(r['round']) for r in table], [1, 2, 3])
            for r, m in zip(table, log):
                self.assertAlmostEqual(float(r['accuracy']), m['accuracy'])
                self.assertAlmostEqual(float(r['old_acc']), m['old_acc'])
                self.assertAlmostEqual(float(r['mcc']), 1.0)
                self.assertEqual(int(r['upload_bytes']), m['bytes']['upload'])

    def test_cluster_cli_and_plot(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in ('none', 'plain'):
                res = cli('secure_code_cluster', '--smoke', '--cluster', mode, '--agg', 'plain', '--rounds', '2',
                          '--no-progress', '--output', str(Path(d) / mode))
                self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            p = cli('tests.plot_cluster', '--runs', str(Path(d) / 'none'), str(Path(d) / 'plain'), '--out', d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            with (Path(d) / 'cluster_summary.csv').open() as f:
                table = {r['mode']: r for r in csv.DictReader(f)}
            for mode in ('none', 'plain'):
                log = rows(Path(d) / mode / 'metrics.jsonl')
                self.assertAlmostEqual(float(table[mode]['final_acc']), log[-1]['accuracy'])
                self.assertAlmostEqual(float(table[mode]['best_acc']), max(r['accuracy'] for r in log))
                self.assertEqual(float(table[mode]['union_mcc']), 1.0)
            self.assertEqual((table['none']['groups'], table['plain']['groups']), ('1', '2'))

    def test_plot_reads_old_format_rows(self):
        from tests.plot_secure import plot
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d) / 'old'
            run_dir.mkdir()
            (run_dir / 'metrics.jsonl').write_text(json.dumps(dict(round=1, accuracy=.5, seconds={}, bytes={})) + '\n')
            plot([run_dir], Path(d))
            self.assertTrue((Path(d) / 'secure.csv').exists())


if __name__ == '__main__':
    unittest.main()
