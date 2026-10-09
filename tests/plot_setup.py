"""Paper figure of the setup sweep (setup_smoke_hybrid output): computation and communication of the
similar (fuzzy) label union vs the number of clients, for one or more runs (e.g. 1 and 16 pair workers).

  python -m tests.plot_setup runs/A/setup_full/setup.json:"1 worker" runs/B/setup_full/setup.json:"16 workers" \
      --out runs/setup_fig

(a) computation: measured MPC time on the benchmark host per run (all 2-party matching sessions + global
    grouping + padding MPC), and the per-client latency of a deployment where every client is its own host
    and disjoint pairs run at once (round-robin: n-1 rounds of one 2-party session, from the 1-worker run:
    uncontended session time) + grouping
(b) communication: traffic per client (sent + received, identical in every run), bars
(c) communication time per client (bytes / bandwidth + round trips x RTT; a run with w pair workers overlaps
    the round trips of min(w, n - 1) of a client's pair sessions), plain union vs each run, bars
The plaintext union (no privacy) is drawn for reference. Writes setup.{png,pdf} and setup_table.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#8172B3']


def load(spec):
    path, _, label = spec.partition(':')
    report = json.loads(Path(path).read_text())
    secure = {r['clients']: r for r in report['results'] if r['method'] != 'plain' and 'deploy_seconds' in r}
    plain = {r['clients']: r['setup_wall_seconds'] for r in report['results'] if r['method'] == 'plain'}
    plain_net = {r['clients']: r.get('comm_seconds_sequential') for r in report['results'] if r['method'] == 'plain'}
    if not secure:
        raise SystemExit(f'{path}: no secure rows with timings (re-run the setup with this version)')
    first = next(iter(secure.values()))
    label = label or f"{first['pair_workers']} pair worker{'s' * (first['pair_workers'] > 1)}"
    return label, secure, plain, plain_net


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='setup.json[:label]')
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    runs = [load(r) for r in a.runs]
    out = a.out or Path(a.runs[0].partition(':')[0]).parent
    out.mkdir(parents=True, exist_ok=True)
    n = sorted(set.intersection(*(set(s) for _, s, _, _ in runs)))
    base = min(runs, key=lambda r: next(iter(r[1].values()))['pair_workers'])   # least contended run
    measured = next(iter(base[1].values()))['backend'] != 'mpc-model'
    ref = base[1]
    minutes = lambda v: v / 60
    xi = list(range(len(n)))                                              # same categorical axis in (a) and (b)

    plain = base[2]
    mbps, rtt = ref[n[0]]['net_mbps'], ref[n[0]]['net_rtt_ms']

    def bars(axis, groups, log=False, fmt=lambda v: f'{v:.2g}'):
        """Grouped bars, one group per client count, value on every bar."""
        width = .84 / len(groups) if len(groups) > 1 else .6
        for j, (label, vals, color, hatch) in enumerate(groups):
            pos = [i + (j - (len(groups) - 1) / 2) * width for i in xi]
            axis.bar(pos, vals, width, color=color, hatch=hatch, edgecolor='white' if hatch else None,
                     linewidth=0, label=label)
            for x_, v in zip(pos, vals):
                axis.text(x_, v, fmt(v), ha='center', va='bottom', fontsize=5 if len(groups) > 1 else 6,
                          rotation=90 if len(groups) > 1 else 0)
        axis.set_xticks(xi, [str(k) for k in n])
        axis.set_xlabel('clients (n / 5 per dataset)')
        if log:
            axis.set_yscale('log')
            lo, hi = axis.get_ylim()
            axis.set_ylim(lo, hi * 30)                                       # room for labels + legend
        else:
            axis.margins(y=.25)
        axis.spines[['top', 'right']].set_visible(False)

    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': .6, 'hatch.linewidth': .5})
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.16, 2.4))             # IEEE double column (figure*)
    num = lambda v: '<0.01' if v < .01 else f'{v:.2f}' if v < 1 else f'{v:.1f}' if v < 10 else f'{v:.0f}'

    # (a) computation: plain union, every run on one host, per-client latency with own hosts
    groups = [('plain union', [minutes(plain.get(k) or 0) for k in n], 'grey', None)]
    groups += [(f'one host, {label}', [minutes(s[k]['host_compute_seconds']) for k in n], COLORS[i % 4], None)
               for i, (label, s, _, _) in enumerate(runs)]
    groups += [('per client, own host', [minutes(ref[k]['deploy_compute_seconds']) for k in n], 'white', '////')]
    bars(ax, groups, log=True, fmt=num)
    for patch in ax.patches[-len(n):]:                                       # outlined hatched bars
        patch.set_edgecolor('k')
        patch.set_linewidth(.5)
    ax.set_ylabel('time (min)')
    ax.set_title(f"(a) MPC computation ({'measured' if measured else 'modelled'})", fontsize=7.5)
    ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2, columnspacing=.8, handlelength=1.2)

    # (b) traffic per client: identical in every run
    bars(bx, [('similar (MPC)', [ref[k]['comm_MB_per_client_max'] / 1e3 for k in n], COLORS[2], None)],
         fmt=lambda v: f'{v:.2f}' if v < 10 else f'{v:.1f}')
    bx.set_ylabel('traffic per client (GB)')
    bx.set_title('(b) communication size', fontsize=7.5)

    # (c) network time per client: plain union vs each run (its pair sessions' round trips overlap)
    groups = [('plain union', [minutes(base[3].get(k) or 0) for k in n], 'grey', None)]
    groups += [(label, [minutes(s[k]['comm_seconds_workers']) for k in n], COLORS[i % 4], None)
               for i, (label, s, _, _) in enumerate(runs)]
    bars(cx, groups, fmt=num)
    cx.set_ylabel('time (min)')
    cx.set_title('(c) communication time', fontsize=7.5)
    cx.legend(fontsize=5.5, frameon=False, loc='upper left', title=f'modelled: {mbps:g} Mbit/s, {rtt:g} ms RTT',
              title_fontsize=5.5, alignment='left', handlelength=1.2)
    fig.tight_layout(w_pad=1.2)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'setup.{ext}', dpi=300)

    with (out / 'setup_table.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clients', 'labels_union', 'pair_mcc']
                   + [f'host_min[{label}]' for label, _, _, _ in runs]
                   + [f'net_min[{label}]' for label, _, _, _ in runs] + ['net_min[plain]']
                   + ['speedup' if len(runs) > 1 else None][:len(runs) > 1]
                   + ['pair_session_s', 'per_client_compute_min', 'traffic_per_client_GB', 'traffic_total_GB',
                      'plain_union_s'])
        for k in n:
            r = ref[k]
            host = [minutes(s[k]['host_compute_seconds']) for _, s, _, _ in runs]
            nets = [round(minutes(s[k]['comm_seconds_workers']), 2) for _, s, _, _ in runs] + [
                round(minutes(base[3].get(k) or 0), 3)]
            w.writerow([k, r.get('union_size'), r.get('pair_mcc')] + [round(h, 2) for h in host] + nets
                       + ([round(max(host) / min(host), 2)] if len(runs) > 1 else [])
                       + [round(r['pair_session_seconds'], 3), round(minutes(r['deploy_compute_seconds']), 2),
                          round(r['comm_MB_per_client_max'] / 1e3, 3),
                          round(((r.get('mpc_measured') or {}).get('global_MB') or 0) / 1e3, 1),
                          plain.get(k)])
    print(f'{out}/setup.png, setup.pdf, setup_table.csv')


if __name__ == '__main__':
    main()
