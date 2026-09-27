"""Run every mapping method with the same settings and plot accuracy per round.

From the project root:
  python -m tests.plot_psi_acc --smoke --rounds 10               # all methods, CPU, synthetic
  python -m tests.plot_psi_acc --rounds 45 --mapping-round 25 --device cuda:0
  python -m tests.plot_psi_acc --mappings image_bi fuzzy_psi_circuit --rounds 10
  python -m tests.plot_psi_acc --device cuda:0    # real data (config.yaml)
  python -m tests.plot_psi_acc --runs DIR [DIR..] # only plot existing runs
Unknown arguments go to main (e.g. --rounds 10 --mapping-round 5).
No PACFL by default (--clustering none); pass --clustering pacfl to override.
Writes <out>/psi_acc.png and <out>/psi_acc.csv.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAPPINGS = ('by_class', 'image_bi', 'psi_trivial_circuit', 'fuzzy_psi_circuit')
LABELS = {'by_class': 'by_class (oracle)', 'image_bi': 'image_bi',
          'psi_trivial_circuit': 'exact PSI', 'fuzzy_psi_circuit': 'fuzzy PSI'}
KEYED = ('psi_trivial_circuit_with_key', 'fuzzy_psi_circuit_with_key')   # same mapping + key release
LABELS.update({'psi_trivial_circuit_with_key': 'exact PSI + key', 'fuzzy_psi_circuit_with_key': 'fuzzy PSI + key'})
PANELS = (('ground_truth_acc', 'Global acc (ground truth)'), ('old_acc', 'Global acc (old)'),
          ('f1', 'Mapping F1'))


def load(run):
    mapping = json.loads((run / 'config.json').read_text())['mapping_strategy']
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line]
    return LABELS.get(mapping, mapping), [r for r in rows if 'evaluation' in r]


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.5 * len(PANELS), 4))
    table = []
    styles = [('o', '-'), ('s', '--'), ('^', '-.'), ('D', ':')]
    for k, run in enumerate(runs):
        label, rows = load(run)
        if not rows:
            raise ValueError(f'{run}: no evaluated rounds (rounds < mapping round?)')
        rounds = [r['round'] for r in rows]
        values = {'ground_truth_acc': [r['evaluation']['ground_truth_acc'] for r in rows],
                  'old_acc': [r['evaluation']['old_acc'] for r in rows],
                  'f1': [r['mapping_metrics']['f1'] for r in rows]}
        # Identical curves would hide each other: dodge x slightly, vary marker and line.
        shift = (k - (len(runs) - 1) / 2) * .08
        marker, line = styles[k % len(styles)]
        for ax, (key, title) in zip(axes, PANELS):
            ax.plot([n + shift for n in rounds], values[key], marker=marker, linestyle=line,
                    markersize=6, alpha=.85, label=label)
        table += [[label, n, *(values[k][i] for k, _ in PANELS)] for i, n in enumerate(rounds)]
    for ax, (_, title) in zip(axes, PANELS):
        ax.set(title=title, xlabel='Round', ylim=(0, 1.02))
        ax.grid(alpha=.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(out / 'psi_acc.png', dpi=150)
    with (out / 'psi_acc.csv').open('w', newline='') as f:
        csv.writer(f).writerows([['method', 'round', *(k for k, _ in PANELS)], *table])
    return out / 'psi_acc.png'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', type=Path, help='plot existing run directories only')
    p.add_argument('--out', type=Path)
    p.add_argument('--mappings', nargs='+', choices=MAPPINGS + KEYED, default=MAPPINGS)
    args, forwarded = p.parse_known_args()
    out = args.out or Path('runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_all_acc')
    out.mkdir(parents=True, exist_ok=True)
    if '--clustering' not in forwarded:
        forwarded += ['--clustering', 'none']  # no PACFL: one generator per client
    runs = args.runs
    if not runs:
        runs = [out / m for m in args.mappings]
        for mapping, run in zip(args.mappings, runs):
            subprocess.run([sys.executable, '-m', 'main', '--mapping', mapping,
                            '--output', str(run), *forwarded], check=True)
    print(f'Plot: {plot(runs, out).resolve()}')


if __name__ == '__main__':
    main()
