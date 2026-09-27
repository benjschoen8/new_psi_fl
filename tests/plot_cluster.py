"""P5: no-cluster vs private PACFL vs plaintext groupings, same data and settings.

From the project root:
  python -m tests.plot_cluster --smoke --rounds 10
  python -m tests.plot_cluster --rounds 45 --tau .5 --devices cuda:0 --workers 4
  python -m tests.plot_cluster --modes none private --rounds 20
  python -m tests.plot_cluster --runs DIR [DIR..]          # only plot existing runs
A crashed run resumes from its checkpoint when the same command is rerun with the same --out.
Unknown arguments go to secure_code_cluster (--rounds, --tau, --agg, --proj-dim, ...).
Writes <out>/cluster_acc.png (accuracy per round) and <out>/cluster_summary.csv
(groups, ARI vs plain grouping, preprocessing seconds / AND gates / word OTs, upload bytes, final acc).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

MODES = ('none', 'original', 'plain', 'private')
LABELS = {'none': 'no cluster', 'original': 'PACFL (angles, plain)', 'plain': 'projected PACFL (plain)',
          'private': 'private PACFL'}


def load(run):
    """setup.json (written at the start of a run) and one metrics row per round (last one wins)."""
    setup = json.loads((run / 'setup.json').read_text())
    rows = {}
    for line in (run / 'metrics.jsonl').read_text().splitlines():
        if line:
            row = json.loads(line)
            rows[row['round']] = row
    if not rows:
        raise ValueError(f'{run}: no rounds in metrics.jsonl')
    return setup, [rows[r] for r in sorted(rows)]


def summary(setup, rows):
    st = setup.get('grouping_stats', {})
    um = setup.get('union_metrics') or rows[-1].get('union_metrics') or {}
    return dict(mode=setup['cluster'], groups=setup['groups'], rounds=rows[-1]['round'],
                ari_vs_plain=setup.get('experimenter_view', {}).get('agreement_with_plain'),
                union_mcc=um.get('mcc'), union_f1=um.get('f1'),
                preprocessing_s=round(st.get('pairs_seconds', 0) + st.get('mpc_seconds', 0), 2),
                word_ots=st.get('pairs_word_ots', 0), and_gates=st.get('pairs_and_gates', 0) + st.get('mpc_and_gates', 0),
                upload_bytes_per_round=rows[-1]['bytes']['upload'],
                final_acc=rows[-1]['accuracy'], best_acc=max(r['accuracy'] for r in rows),
                final_old_acc=rows[-1].get('old_acc'))


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    styles = [('o', '-'), ('s', '--'), ('^', '-.'), ('D', ':')]
    table = []
    for k, run in enumerate(runs):
        setup, rows = load(run)
        marker, line = styles[k % len(styles)]
        shift = (k - (len(runs) - 1) / 2) * .08                         # identical curves stay visible
        label = f"{LABELS.get(setup['cluster'], setup['cluster'])} (G={setup['groups']})"
        xs = [r['round'] + shift for r in rows]
        ax.plot(xs, [r['accuracy'] for r in rows], marker=marker, linestyle=line, color=f'C{k}',
                markersize=5, alpha=.85, label=label)
        ax2.plot(xs, [r.get('old_acc', float('nan')) for r in rows], marker=marker, linestyle=line, color=f'C{k}',
                 markersize=5, alpha=.85, label=label)
        table.append(summary(setup, rows))
    ax.set(title='Global accuracy (ground truth)', xlabel='Round', ylabel='Accuracy')
    ax2.set(title='Global accuracy (old: predicted mapping)', xlabel='Round')
    for a in (ax, ax2):
        a.grid(alpha=.3)
        a.set_ylim(bottom=0)                                            # auto top: small values visible
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'cluster_acc.png', dpi=150)
    plt.close(fig)
    with (out / 'cluster_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return out / 'cluster_acc.png', table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', type=Path, help='plot existing run directories only')
    p.add_argument('--out', type=Path)
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    args, forwarded = p.parse_known_args()
    out = args.out or Path('runs') / (time.strftime('%Y%m%dT%H%M%S') + '_cluster_compare')
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs
    if not runs:
        runs = [out / m for m in args.modes]
        for mode, run in zip(args.modes, runs):
            cmd = [sys.executable, '-m', 'secure_code_cluster', '--cluster', mode, '--output', str(run),
                   '--no-progress', *forwarded]
            if (run / 'checkpoint_last.pt').exists():                   # rerun of a crashed comparison: resume
                cmd += ['--resume', str(run / 'checkpoint_last.pt')]
            if subprocess.run(cmd).returncode:                           # one failed mode must not lose the others
                print(f'[plot_cluster] {mode} failed; rerun the same command to resume it', file=sys.stderr)
    runs = [r for r in runs if (r / 'metrics.jsonl').exists() and (r / 'setup.json').exists()]
    if not runs:
        raise SystemExit('no finished runs to plot')
    png, table = plot(runs, out)
    for row in table:
        print(row)
    print(f'Plot: {png.resolve()}')


if __name__ == '__main__':
    main()
