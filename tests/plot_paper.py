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
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE,
                           gridspec_kw=dict(width_ratios=[1, 1.45, 1], wspace=.3))
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=.8)
        a.set_axisbelow(True)

    a = ax[0]                                                            # (a) accuracy
    for d, lab, c in zip(data, labels, COLORS):
        a.plot(d['rounds'], d['acc'], color=c, linewidth=2, marker='o', markersize=5,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    ends = sorted(((d['acc'][-1], d['rounds'][-1], c) for d, c in zip(data, COLORS)), key=lambda e: e[0])
    gap, prev = .045 * max(max(d['acc']) for d in data), -1.
    dx = .03 * max(max(d['rounds']) - min(d['rounds']), 1)
    for acc, rnd, c in ends:                                             # final values, nudged apart
        y_ = max(acc, prev + gap)
        a.text(rnd + dx, y_, f'{acc:.3f}', va='center', color=c, fontsize=8, fontweight='bold', clip_on=False)
        prev = y_
    a.set(title='(a) Global accuracy', xlabel='Round', ylabel='Ground-truth accuracy')
    a.set_xticks(data[0]['rounds'])
    a.set_ylim(bottom=0)
    a.grid(axis='x', visible=False)

    a = ax[1]                                                            # (b) time per stage
    x = np.arange(len(STAGES))
    w = .8 / len(data)
    top = max(max(d['seconds'].values()) for d in data)
    for i, (d, lab, c) in enumerate(zip(data, labels, COLORS)):
        v = [d['seconds'][k] for k, _ in STAGES]
        bars = a.bar(x + (i - (len(data) - 1) / 2) * w, v, width=w - .04, color=c, label=lab,
                     edgecolor=SURFACE, linewidth=1)
        for b_, t in zip(bars, v):
            a.text(b_.get_x() + b_.get_width() / 2, t + top * .015, f'{t:.0f}' if t >= 10 else f'{t:.1f}',
                   ha='center', va='bottom', color=INK2, fontsize=7, rotation=90)
    a.set_xticks(x, [n.replace(' ', '\n', 1) for _, n in STAGES], fontsize=8)
    a.set(title='(b) Mean time per round', ylabel='Seconds')
    a.grid(axis='x', visible=False)
    a.set_ylim(top=top * 1.22)

    a = ax[2]                                                            # (c) communication
    groups = ['Upload', 'Download']
    x = np.arange(len(groups))
    w = .8 / len(data)
    for i, (d, lab, c) in enumerate(zip(data, labels, COLORS)):
        v = [d['upload'] / 1e6, d['download'] / 1e6]
        bars = a.bar(x + (i - (len(data) - 1) / 2) * w, v, width=w - .04, color=c, label=lab,
                     edgecolor=SURFACE, linewidth=1)
        for b_, raw in zip(bars, [d['upload'], d['download']]):
            a.text(b_.get_x() + b_.get_width() / 2, b_.get_height() * 1.12, human(raw), ha='center',
                   va='bottom', color=INK2, fontsize=7, rotation=90)
    a.set_xticks(x, groups)
    a.set_yscale('log')                                                  # kB uploads next to MB downloads
    lo = min(min(d['upload'], d['download']) for d in data if d['upload'] or d['download']) / 1e6
    hi = max(max(d['upload'], d['download']) for d in data) / 1e6
    a.set_ylim(lo / 3, hi * 12)
    a.set(title='(c) Traffic per client per round', ylabel='MB (log scale)')
    a.grid(axis='x', visible=False)

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
