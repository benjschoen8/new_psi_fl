"""Paper figure: Plain-GeFL vs ours, from run folders of secure_code_no_cluster (--gen cbn).

  (a) global accuracy per round   (b) mean seconds per round by stage   (c) mean bytes per client per round

  python -m tests.plot_paper --runs runs/plain runs/ours --labels Plain-GeFL Ours --out figs
Writes paper.png, paper.pdf and paper.csv (the numbers behind every bar and point).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLORS = ['#eb6834', '#2a78d6', '#1baf7a', '#eda100']          # fixed categorical order (validated)
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
STAGES = [('local_training', 'Local training'), ('aggregation', 'Aggregation'), ('downlink', 'Downlink'),
          ('global_training', 'Global classifier'), ('evaluation', 'Evaluation')]


def load(run):
    rows = {}
    for line in (Path(run) / 'metrics.jsonl').read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r['round']] = r
    rows = [rows[k] for k in sorted(rows)]
    setup = json.loads((Path(run) / 'setup.json').read_text())
    later = [r for r in rows if r['round'] > 1] or rows                 # round 1 has no downlink
    b = lambda r, k: r['bytes'].get(k, r['bytes'].get('upload', 0) if k.startswith('upload') else 0)
    return dict(
        rounds=[r['round'] for r in rows], acc=[r['accuracy'] for r in rows],
        seconds={k: float(np.mean([r['seconds'].get(k, 0) for r in rows])) for k, _ in STAGES},
        upload=float(np.mean([b(r, 'upload_per_client') for r in rows])),
        download=float(np.mean([b(r, 'download_per_client') for r in later])),
        setup_up=setup.get('union', {}).get('setup_upload_bytes_per_client', 0),
        setup_s=setup.get('union', {}).get('seconds', 0))


def human(x):
    for unit, s in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if x >= s:
            return f'{x / s:.2f} {unit}' if x < 10 * s else f'{x / s:.1f} {unit}'
    return f'{x:.0f} B'


def plot(runs, labels, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
                         'axes.titlesize': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    data = [load(r) for r in runs]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=SURFACE,
                           gridspec_kw=dict(width_ratios=[1, 1.25, 1], wspace=.42))
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=.8)
        a.set_axisbelow(True)

    a = ax[0]                                                            # (a) accuracy
    for d, lab, c in zip(data, labels, COLORS):
        a.plot(d['rounds'], d['acc'], color=c, linewidth=2, marker='o', markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        a.annotate(f"{d['acc'][-1]:.3f}", (d['rounds'][-1], d['acc'][-1]), xytext=(6, 0),
                   textcoords='offset points', va='center', color=INK2, fontsize=8)
    a.set(title='(a) Global accuracy', xlabel='Round', ylabel='Ground-truth accuracy')
    a.set_xticks(data[0]['rounds'])
    a.set_ylim(bottom=0)
    a.grid(axis='x', visible=False)

    a = ax[1]                                                            # (b) time per stage
    y = np.arange(len(STAGES))
    h = .8 / len(data)
    for i, (d, lab, c) in enumerate(zip(data, labels, COLORS)):
        v = [d['seconds'][k] for k, _ in STAGES]
        bars = a.barh(y + (i - (len(data) - 1) / 2) * h, v, height=h - .04, color=c, label=lab,
                      edgecolor=SURFACE, linewidth=1)
        for b_, x in zip(bars, v):
            a.text(x + .8, b_.get_y() + b_.get_height() / 2, f'{x:.1f}', va='center', color=INK2, fontsize=7.5)
    a.set_yticks(y, [n for _, n in STAGES])
    a.invert_yaxis()
    a.set(title='(b) Mean time per round', xlabel='Seconds')
    a.grid(axis='y', visible=False)
    a.set_xlim(right=max(max(d['seconds'].values()) for d in data) * 1.18)
    tot = '   '.join(f"{lab}: {sum(d['seconds'].values()):.0f} s/round" for d, lab in zip(data, labels))
    a.text(0, -.2, tot, transform=a.transAxes, color=INK2, fontsize=8)

    a = ax[2]                                                            # (c) communication
    groups = ['Upload', 'Download']
    x = np.arange(len(groups))
    w = .8 / len(data)
    for i, (d, lab, c) in enumerate(zip(data, labels, COLORS)):
        v = [d['upload'] / 1e6, d['download'] / 1e6]
        bars = a.bar(x + (i - (len(data) - 1) / 2) * w, v, width=w - .04, color=c, label=lab,
                     edgecolor=SURFACE, linewidth=1)
        for b_, raw in zip(bars, [d['upload'], d['download']]):
            a.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), human(raw), ha='center', va='bottom',
                   color=INK2, fontsize=7.5)
    a.set_xticks(x, groups)
    a.set(title='(c) Traffic per client per round', ylabel='MB')
    a.grid(axis='x', visible=False)
    setup = '   '.join(f"{lab} setup: {human(d['setup_up'])} up, {d['setup_s']:.0f} s"
                       for d, lab in zip(data, labels) if d['setup_up'])
    if setup:
        a.text(0, -.2, setup, transform=a.transAxes, color=INK2, fontsize=8)

    handles, labs = ax[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, 1.02))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'paper.{ext}', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    with (out / 'paper.csv').open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['method', 'round', 'accuracy'])
        for d, lab in zip(data, labels):
            wr.writerows([lab, r, a_] for r, a_ in zip(d['rounds'], d['acc']))
        wr.writerow([])
        wr.writerow(['method', *[k for k, _ in STAGES], 'upload_per_client', 'download_per_client',
                     'setup_upload_per_client', 'setup_seconds'])
        for d, lab in zip(data, labels):
            wr.writerow([lab, *[d['seconds'][k] for k, _ in STAGES], d['upload'], d['download'],
                         d['setup_up'], d['setup_s']])
    return out / 'paper.png'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--out', default='.')
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]
    if len(labels) != len(a.runs):
        raise SystemExit('--labels needs one name per run')
    print(plot(a.runs, labels, a.out))


if __name__ == '__main__':
    main()
