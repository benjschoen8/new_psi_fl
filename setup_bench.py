"""Setup timing experiments (similar = fuzzy union, real MPC, one thread per client), each point run REPEATS
times in a FRESH process, results averaged, a log per run and a matplotlib figure per experiment.

  clients   5 / 10 / 30 / 50 clients, 5 datasets (MNIST, EMNIST, CIFAR10, FashionMNIST, STL10)
  datasets  30 clients, 3 / 5 / 7 datasets (3: MNIST, EMNIST, CIFAR10; 7: + CIFAR100, USPS)
  estimate  100 clients x 100 labels each: cost model of the same circuits, calibrated by the measured runs

Every client holds all labels of one dataset (clients split evenly over the datasets), every client pads to
--pad-to labels (public policy, default 100), pair sessions: one per client at a time, all clients at once
(--pair-workers 0). Between runs: the MPC processes of the finished run are checked (leftovers killed), the
run's process has exited (memory, threads, sockets released by the OS), and free memory is logged.

  python -m setup_bench                         # clients + datasets, real MPC, 3 runs per point
  python -m setup_bench --only big              # 100 x 100, real MPC, separately (memory check first)
  python -m setup_bench --only estimate         # 100 x 100 from the cost model (no MPC)
  python -m setup_bench --only clients --repeats 1
  python -m setup_bench --model --repeats 1     # cost model, minutes: check MCC / U before the real run
Output: <out>/<experiment>/<point>/run<k>/{run.log, setup.json}, <out>/<experiment>/summary.{csv,json},
<out>/<experiment>/time.{png,pdf}, <out>/estimate.json, <out>/bench.log
"""
import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

FIVE = 'MNIST,EMNIST,CIFAR10,FashionMNIST,STL10'
DATASETS = {3: 'MNIST,EMNIST,CIFAR10', 5: FIVE, 7: FIVE + ',CIFAR100,USPS'}   # no SVHN
MPC_NAMES = ('party.x',)                    # MP-SPDZ executables: *-party.x


def mem_available_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float('nan')


def mpc_processes():
    """(pid, name) of running MP-SPDZ parties (matched on the executable name, not the command line)."""
    out = subprocess.run(['ps', '-eo', 'pid=,comm='], capture_output=True, text=True).stdout
    procs = []
    for line in out.splitlines():
        pid, _, name = line.strip().partition(' ')
        if name.strip().endswith(MPC_NAMES) and int(pid) != os.getpid():
            procs.append((int(pid), name.strip()))
    return procs


def clean(log):
    """Before / after every run: no MP-SPDZ party may survive (it would hold RAM, CPU and ports)."""
    left = mpc_processes()
    for pid, name in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if left:
        log(f'  cleanup: killed {len(left)} leftover MPC processes ({sorted({n for _, n in left})})')
        time.sleep(2)
    if mpc_processes():
        raise RuntimeError('MPC processes survive SIGKILL; stop and check the machine')
    time.sleep(3)                                                # sockets in TIME_WAIT, page cache settle


def run_point(args, exp, point, datasets, clients, log, extra=()):
    rows = []
    for k in range(1, args.repeats + 1):
        d = args.out / exp / point / f'run{k}'
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'setup.json').exists() and not args.rerun:
            log(f'{exp} {point} run {k}: done before, reused')
        else:
            clean(log)
            before = mem_available_gb()
            cmd = [sys.executable, '-m', 'setup_smoke_hybrid', '--datasets', datasets, '--clients', str(clients),
                   '--class-subsets', '100,100', '--class-share', 'full', '--num-train-cifar10stl10', '0',
                   '--methods', 'plain', 'similar', '--pair-protocol', 'hegc', '--pca-dim', str(args.pca_dim),
                   '--gc-protocol', args.gc_protocol, '--group-protocol', 'atlas', '--group-version', '2',
                   '--pad-to', str(args.pad_to), '--pair-workers', '0', '--mpc-timeout', str(args.mpc_timeout),
                   '--net-mbps', str(args.net_mbps), '--net-rtt-ms', str(args.net_rtt_ms),
                   '--seed', str(args.seed), '--repeats', '1', '--out', str(d)] + (['--mpc-model'] if args.model else [])
            cmd += list(extra)
            log(f'{exp} {point} run {k}/{args.repeats}: start, {clients} clients, datasets {datasets}, '
                f'free RAM {before:.1f} GB')
            t0 = time.perf_counter()
            with open(d / 'run.log', 'w') as f:                  # fresh process: memory / threads freed on exit
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
            clean(log)
            log(f'{exp} {point} run {k}: exit {rc} after {time.perf_counter() - t0:.0f} s, '
                f'free RAM {mem_available_gb():.1f} GB (before {before:.1f} GB)')
            if rc:
                raise SystemExit(f'run failed, see {d / "run.log"}')
        res = json.loads((d / 'setup.json').read_text())['results']
        sec = next(r for r in res if r['method'] != 'plain')
        plain = next(r for r in res if r['method'] == 'plain')
        mpc = sec.get('mpc_measured') or {}
        rows.append(dict(mpc_s=sec['host_compute_seconds'], matching_s=mpc.get('matching_seconds') or 0.,
                         grouping_s=(mpc.get('group_seconds') or 0.) + (mpc.get('pad_max_seconds') or 0.),
                         setup_s=sec['setup_wall_seconds'], plain_s=plain['setup_wall_seconds'],
                         traffic_GB=sec['comm_MB_per_client_max'] / 1e3,
                         total_GB=(mpc.get('global_MB') or 0) / 1e3, network_s=sec['comm_seconds_model_max'],
                         mcc=sec.get('pair_mcc'), U=sec.get('union_size'), plain_U=plain.get('union_size'),
                         labels=sec['labels_per_client'], padded=sec.get('padded_to'),
                         model_s=_model(clients, args, mpc)))
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    std = {k: math.sqrt(sum((r[k] - mean[k]) ** 2 for r in rows) / max(1, len(rows) - 1)) for k in mean}
    log(f'{exp} {point}: MPC {mean["mpc_s"]:.1f} +- {std["mpc_s"]:.1f} s (matching {mean["matching_s"]:.1f}, '
        f'grouping {mean["grouping_s"]:.1f}), {mean["traffic_GB"]:.2f} GB/client, U {rows[0]["U"]} '
        f'(plain {rows[0]["plain_U"]}), MCC {rows[0]["mcc"]}')
    return dict(point=point, clients=clients, datasets=datasets, runs=len(rows), mean=mean, std=std, rows=rows)


def _model(n, args, mpc):
    """Cost-model MPC seconds of the same configuration on this host (for the calibration)."""
    from label_union.mpc_model import estimate
    e = estimate(n, args.pad_to, args.pca_dim, 45, mpc.get('propagation_steps') or 1, args.gc_protocol,
                 1, max(1, n // 2), 'plain')
    return e['matching_seconds'] + e['group_seconds']


def summary(exp, points, xlabel, out, log):
    d = out / exp
    with (d / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        keys = ['mpc_s', 'matching_s', 'grouping_s', 'network_s', 'traffic_GB', 'total_GB', 'plain_s']
        w.writerow(['point', 'clients', 'datasets', 'runs', 'U', 'mcc']
                   + [f'{k}_{s}' for k in keys for s in ('mean', 'std')])
        for p in points:
            w.writerow([p['point'], p['clients'], p['datasets'], p['runs'], p['rows'][0]['U'], p['rows'][0]['mcc']]
                       + [round(p[s][k], 3) for k in keys for s in ('mean', 'std')])
    (d / 'summary.json').write_text(json.dumps(points, indent=2) + '\n')
    plot(points, xlabel, d, exp)
    log(f'{exp}: {d / "summary.csv"}, {d / "time.png"}')


def plot(points, xlabel, d, exp, est=None):
    """est: the 100 x 100 estimate, drawn as a hatched extra point (clients experiment)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': .6, 'hatch.linewidth': .5})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))
    x = list(range(len(points)))
    names = [p['point'] for p in points]
    m = lambda k: [p['mean'][k] / 60 for p in points]
    e = lambda k: [p['std'][k] / 60 for p in points]
    runs = points[0]['runs']
    ax, bx, cx = axes
    ax.bar(x, m('matching_s'), .6, color='#4C72B0', label='pairwise matching (2PC)')
    ax.bar(x, m('grouping_s'), .6, bottom=m('matching_s'), color='#DD8452', label='global grouping (MPC)',
           yerr=e('mpc_s'), capsize=2, error_kw=dict(lw=.6))
    for i, v in enumerate(m('mpc_s')):
        ax.text(i, v, f'{v:.2f}' if v < 10 else f'{v:.0f}', ha='center', va='bottom', fontsize=6)
    ax.plot(x, m('plain_s'), 'k:', marker='x', ms=3, lw=.8, label='plaintext union (no privacy)')
    ax.set_ylabel('time (min)')
    ax.set_title('(a) setup computation, measured', fontsize=7.5)
    ax.legend(fontsize=5.5, frameon=False, loc='upper left')
    bx.bar(x, [p['mean']['traffic_GB'] for p in points], .6, color='#55A868')
    for i, p in enumerate(points):
        v = p['mean']['traffic_GB']
        bx.text(i, v, f'{v:.2f}' if v < 10 else f'{v:.1f}', ha='center', va='bottom', fontsize=6)
    bx.set_ylabel('traffic per client (GB)')
    bx.set_title('(b) communication size', fontsize=7.5)
    cx.bar(x, m('network_s'), .6, color='#8172B3')
    for i, v in enumerate(m('network_s')):
        cx.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=6)
    cx.set_ylabel('time (min)')
    cx.set_title('(c) communication time, modelled', fontsize=7.5)
    if est:                                                      # 100 x 100 estimate, hatched, after the measured
        i = len(points)
        names = names + [f"{est['clients']}\n(est.)"]
        kw = dict(width=.6, hatch='////', edgecolor='k', linewidth=.4)
        ax.bar(i, est['matching_min'], color='#4C72B0', **kw)
        ax.bar(i, est['grouping_min'], bottom=est['matching_min'], color='#DD8452', **kw)
        bx.bar(i, est['traffic_per_client_GB'], color='#55A868', **kw)
        cx.bar(i, est['network_min'], color='#8172B3', **kw)
        for axis, v in ((ax, est['mpc_min']), (bx, est['traffic_per_client_GB']), (cx, est['network_min'])):
            axis.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=6)
        x = x + [i]
    for a in axes:
        a.set_xticks(x, names)
        a.set_xlabel(xlabel)
        a.margins(y=.2)
        a.spines[['top', 'right']].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.5)                                    # room for the legend
    fig.suptitle(f'one thread per client, padded to {points[0]["rows"][0]["padded"]} labels, mean of {runs} '
                 f'run{"s" * (runs > 1)} (bars: +-1 std)', fontsize=7)
    fig.tight_layout(w_pad=1.2)
    for ext in ('png', 'pdf'):
        fig.savefig(d / f'time.{ext}', dpi=300)
    plt.close(fig)


def _calibration(points):
    """measured / model of the largest point whose n MPC processes fit on this host's cores (no contention):
    a deployment gives every client its own host, so host contention must not enter the estimate."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    fit = [p for p in points if p['clients'] <= cores] or points[:1]
    return fit[-1]['mean']['mpc_s'] / fit[-1]['mean']['model_s']


def estimate_100(args, calib, log):
    """100 clients x 100 labels (every client the same, padded to 100): cost model x measured/model ratio."""
    from label_union.mpc_model import estimate, group_memory
    n, m = 100, args.pad_to
    e = estimate(n, m, args.pca_dim, 45, 1, args.gc_protocol, 1, n // 2, 'plain')
    ratio = calib or 1.
    mb = max(s + r for s, r in zip(e['client_sent_MB'], e['client_received_MB']))
    rounds = max(e['client_rounds'])
    res = dict(clients=n, labels_per_client=m, calibration_ratio=ratio,
               matching_min=e['matching_seconds'] * ratio / 60, grouping_min=e['group_seconds'] * ratio / 60,
               mpc_min=(e['matching_seconds'] + e['group_seconds']) * ratio / 60,
               traffic_per_client_GB=mb / 1e3, total_TB=e['global_MB'] / 1e6,
               network_min=(mb * 8 / args.net_mbps + rounds * args.net_rtt_ms / 1e3) / 60,
               grouping_memory_per_client_GB=group_memory(n, m) / 1e3,
               grouping_memory_one_host_GB=group_memory(n, m) * n / 1e3,
               note='estimate: cost model fitted to MP-SPDZ sessions (n <= 50), scaled by the measured/model '
                    'ratio of the largest measured run without core contention; one thread per client, each '
                    'client on its own host')
    (args.out / 'estimate.json').write_text(json.dumps(res, indent=2) + '\n')
    log('estimate 100 clients x 100 labels: ' + ', '.join(f'{k} {v:.1f}' if isinstance(v, float) else f'{k} {v}'
                                                         for k, v in res.items() if k != 'note'))
    return res


def big(args, log):
    """Real MPC, n clients with the same full label set (all classes of --big-dataset, padded to --pad-to).
    The grouping keeps a dense (n * pad)^2 share matrix in every party: check the host's memory first."""
    from label_union.mpc_model import estimate, group_memory
    n, m = args.big_clients, args.pad_to
    need = group_memory(n, m, args.big_block_rows) * n / 1e3 + 4                 # + data, Python, OS
    have = mem_available_gb()
    e = estimate(n, m, args.pca_dim, 45, 1, args.gc_protocol, 1, n // 2, 'plain')
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    hours = ((e['pair_global_MB'] * .02 / cores) + e['group_seconds'] * max(1., n / cores)) * 1.45 / 3600
    log(f'big: {n} clients x {m} labels ({args.big_dataset}), {e["pair_sessions"]} pair sessions, '
        f'grouping {e["group_global_MB"] / n / 1e3:.0f} GB MPC traffic per party; needs ~{need:.0f} GB RAM '
        f'(free {have:.0f} GB), ~{hours:.1f} h per run on {cores} cores (model estimate)')
    if need > have and not args.force:
        raise SystemExit(f'big: ~{need:.0f} GB RAM needed, {have:.0f} GB free: run it on a larger machine '
                         f'(or --force to try anyway)')
    pts = [run_point(argparse.Namespace(**{**vars(args), 'repeats': args.big_repeats,          # the grouping
                                           'mpc_timeout': max(args.mpc_timeout, 172800)}), 'big',  # session: hours
                     f'{n}x{m}', args.big_dataset, n, log,
                     extra=('--group-block-rows', str(args.big_block_rows)))]
    summary('big', pts, f'clients ({args.big_dataset}, all classes each)', args.out, log)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--only', nargs='+', choices=('clients', 'datasets', 'estimate', 'big'),
                   default=['clients', 'datasets'],
                   help='big (separate, real MPC): 100 clients x 100 labels, every client all CIFAR-100 classes')
    p.add_argument('--big-clients', type=int, default=100)
    p.add_argument('--big-dataset', default='CIFAR100', help='every big-run client holds all its classes')
    p.add_argument('--big-repeats', type=int, default=1)
    p.add_argument('--big-block-rows', type=int, default=16, help='grouping work-space rows (memory)')
    p.add_argument('--force', action='store_true', help='big: run even if the memory check says it will not fit')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--clients', type=int, nargs='+', default=[5, 10, 30, 50])
    p.add_argument('--dataset-counts', type=int, nargs='+', default=[3, 5, 7])
    p.add_argument('--dataset-clients', type=int, default=30)
    p.add_argument('--seven', default=None, help='the 7-dataset list (default: the 5 + CIFAR100, USPS)')
    p.add_argument('--pad-to', type=int, default=100)
    p.add_argument('--pca-dim', type=int, default=48)
    p.add_argument('--gc-protocol', default='semi-bin')
    p.add_argument('--mpc-timeout', type=float, default=21600)
    p.add_argument('--net-mbps', type=float, default=100)
    p.add_argument('--net-rtt-ms', type=float, default=20)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--model', action='store_true', help='cost model instead of real MPC (fast check)')
    p.add_argument('--rerun', action='store_true', help='rerun points that already have a setup.json')
    p.add_argument('--out', type=Path, default=Path('runs/setup_bench'))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logf = (args.out / 'bench.log').open('a')

    def log(msg):
        line = f'[{time.strftime("%F %T")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    log(f'start: {vars(args)}')
    calib = None
    if 'clients' in args.only:
        pts = [run_point(args, 'clients', str(n), FIVE, n, log) for n in args.clients]
        summary('clients', pts, 'clients (5 datasets)', args.out, log)
        calib = None if args.model else _calibration(pts)
    if 'datasets' in args.only:
        if args.seven:
            DATASETS[7] = args.seven
        pts = [run_point(args, 'datasets', str(k), DATASETS[k], args.dataset_clients, log)
               for k in args.dataset_counts]
        summary('datasets', pts, f'datasets ({args.dataset_clients} clients)', args.out, log)
    if 'estimate' in args.only:
        if calib is None and (args.out / 'clients' / 'summary.json').exists() and not args.model:
            calib = _calibration(json.loads((args.out / 'clients' / 'summary.json').read_text()))
        est = estimate_100(args, calib, log)
        f = args.out / 'clients' / 'summary.json'
        if f.exists():                                           # clients figure with the estimate added
            plot(json.loads(f.read_text()), 'clients (5 datasets)', args.out / 'clients', 'clients', est)
            log(f'clients figure with the 100 x 100 estimate: {args.out / "clients" / "time.png"}')
    if 'big' in args.only:
        big(args, log)
    log('done')


if __name__ == '__main__':
    main()
