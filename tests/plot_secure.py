"""Run secure_code_no_cluster (plain and/or secagg) and plot accuracy, time per stage, traffic.

From the project root:
  python -m tests.plot_secure --smoke --rounds 20                      # synthetic, seconds
  python -m tests.plot_secure --fast --device cuda:0 \\
      --num-train-mnist 3 --num-train-emnist 0 --num-train-cifar10 3     # small real run
  python -m tests.plot_secure --runs runs/A runs/B                       # plot existing runs only
A crashed run resumes from its checkpoint when the same command is rerun with the same --out.
Unknown arguments go to secure_code_no_cluster. Writes <out>/secure.png and <out>/secure.csv.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

STAGES = ('downlink', 'local_training', 'aggregation', 'global_training', 'evaluation', 'checkpoint')
COLUMNS = ['run', 'round', 'accuracy', 'old_acc', 'mcc', 'f1', *STAGES, 'upload_bytes', 'broadcast_bytes']


def load(run):
    """Rows of metrics.jsonl, one per round (a rerun round after a resume keeps its last row)."""
    rows = {}
    for line in (run / 'metrics.jsonl').read_text().splitlines():
        if line:
            row = json.loads(line)
            rows[row['round']] = row
    um = run / 'evaluator' / 'union_metrics.json'                     # union is scored once (cbn runs)
    if um.exists():
        for row in rows.values():
            row.setdefault('union_metrics', json.loads(um.read_text()))
    args = run / 'args.json'
    a = json.loads(args.read_text()) if args.exists() else {}
    name = f"agg={a['agg']} union={a['union']}" if 'agg' in a and 'union' in a else a.get('agg', run.name)
    return name, [rows[r] for r in sorted(rows)]


def values(row):
    """accuracy = ground-truth accuracy (evaluation.evaluate_global); old runs lack old_acc / mcc."""
    mm = row.get('union_metrics') or {}
    return dict(accuracy=row['accuracy'], old_acc=row.get('old_acc', float('nan')),
                mcc=mm.get('mcc', float('nan')), f1=mm.get('f1', float('nan')))


def table_rows(name, rows):
    return [[name, x['round'], *values(x).values(), *(x['seconds'].get(s, 0) for s in STAGES),
             x['bytes'].get('upload', 0), x['bytes'].get('broadcast', 0)] for x in rows]


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a4, a2, a3) = plt.subplots(1, 4, figsize=(20, 4))
    table, width = [], .8 / max(1, len(runs))
    for k, run in enumerate(runs):
        name, rows = load(run)
        if not rows:
            raise ValueError(f'{run}: no rounds in metrics.jsonl')
        r = [x['round'] for x in rows]
        v = [values(x) for x in rows]
        a1.plot(r, [x['accuracy'] for x in v], marker='os^D'[k % 4], color=f'C{k}', label=f'{name} ground truth')
        a1.plot(r, [x['old_acc'] for x in v], linestyle=':', color=f'C{k}', label=f'{name} old_acc')
        a4.plot(r, [x['mcc'] for x in v], marker='os^D'[k % 4], color=f'C{k}', label=name)
        bottom = 0
        for s in STAGES:
            t = sum(x['seconds'].get(s, 0) for x in rows) / len(rows)
            a2.bar(k, t, bottom=bottom, color=f'C{STAGES.index(s)}', label=s if k == 0 else None)
            bottom += t
        a3.bar([x + k * width for x in r], [x['bytes'].get('upload', 0) / 1e6 for x in rows], width, label=name)
        table += table_rows(name, rows)
    a1.set(title='Global classifier accuracy', xlabel='Round'); a1.set_ylim(bottom=0)   # auto top: low early accuracy stays visible; a1.grid(alpha=.3); a1.legend(fontsize=8)
    a4.set(title='Label union MCC', xlabel='Round', ylim=(-1.05, 1.05)); a4.grid(alpha=.3); a4.legend(fontsize=8)
    a2.set(title='Mean seconds per round by stage', xticks=range(len(runs)),
           xticklabels=[load(r)[0] for r in runs]); a2.legend(fontsize=8)
    a3.set(title='Upload per round (MB, all clients)', xlabel='Round'); a3.legend()
    fig.tight_layout()
    fig.savefig(out / 'secure.png', dpi=150)
    plt.close(fig)
    with (out / 'secure.csv').open('w', newline='') as f:
        csv.writer(f).writerows([COLUMNS, *table])
    return out / 'secure.png'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', type=Path)
    p.add_argument('--aggs', nargs='+', default=['plain', 'secagg'], choices=['plain', 'secagg'])
    p.add_argument('--out', type=Path)
    args, forwarded = p.parse_known_args()
    out = args.out or Path('runs') / (time.strftime('%Y%m%dT%H%M%S') + '_secure_plot')
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs
    if not runs:
        runs = [out / agg for agg in args.aggs]
        for agg, run in zip(args.aggs, runs):
            cmd = [sys.executable, '-m', 'secure_code_no_cluster', '--agg', agg, '--output', str(run), *forwarded]
            if (run / 'checkpoint_last.pt').exists():                   # rerun of a crashed comparison: resume
                cmd += ['--resume', str(run / 'checkpoint_last.pt')]
            if subprocess.run(cmd).returncode:                           # one failed run must not lose the others
                print(f'[plot_secure] {agg} failed; rerun the same command to resume it', file=sys.stderr)
        runs = [r for r in runs if (r / 'metrics.jsonl').exists()]
        if not runs:
            raise SystemExit('no finished runs to plot')
    print(f'Plot: {plot(runs, out).resolve()}')


if __name__ == '__main__':
    main()
