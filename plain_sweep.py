"""Hyper-parameter grid search for max accuracy: Plain-GeFL (--scheme plain, default: no SecAgg, no PSI
cryptography), Ours (--scheme ours: circuit PSI + SecAgg) or Ours fuzzy (--scheme ours_fuzzy).

    python plain_sweep.py                                   # default grid, 3-dataset clients, 15 rounds
    python plain_sweep.py --jobs 3 --rounds 20 --grid '{"--gen-widths": ["128,64,32"], "gen_lr": [2e-4, 5e-4]}'
    python plain_sweep.py --scheme ours --out runs/sweep_ours   # same grid, our scheme
    python plain_sweep.py --summary runs/sweep              # just print the table of a (running) sweep

Grid keys starting with '--' are CLI flags of secure_code_no_cluster ("" value = bare flag, null = absent);
other keys override config.yaml entries. Every combination is one run in <out>/<name>/; finished runs
(DONE) are skipped, so re-running the same command resumes. Ranked by best-round accuracy.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

GRID = {                                     # 3 x 2 x 2 x 2 = 24 runs
    '--gen-widths': ['64,32,16', '128,64,32', '256,128,64'],
    '--warmup-epochs': [0, 10],
    'global_samples_per_class': [64, 512],
    'global_model_epochs': [5, 20],
}
SCHEMES = {'plain': ['--agg', 'plain'], 'ours': [], 'ours_fuzzy': ['--union', 'fuzzy']}   # as run_experiments.sh
DATA = '--num-train-mnist 1 --num-train-emnist 1 --num-train-cifar10 1 --min-holders 1'


def name_of(combo):
    return '_'.join(f"{k.lstrip('-').replace('-', '_')}={str(v).replace(',', '-')}"
                    for k, v in combo.items() if v is not None) or 'default'


def scores(d):
    f = d / 'metrics.jsonl'
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    if not rows:
        return None
    best = max(rows, key=lambda r: r['accuracy'])
    per = {k: round(v['accuracy'], 4) for k, v in best['evaluation'].get('by_dataset', {}).items()}
    bal = sum(per.values()) / len(per) if per else best['accuracy']
    dg = rows[-1].get('diag') or {}                                   # --diagnostics (last round)
    return dict(best=best['accuracy'], best_round=best['round'], last=rows[-1]['accuracy'],
                rounds=rows[-1]['round'], balanced=bal, by_dataset=per,
                **{k: dg.get(k) for k in ('ceiling', 'fidelity_mean', 'synth_fit_mean', 'diversity_mean')})


def summary(out):
    table = []
    for d in sorted(p for p in Path(out).iterdir() if p.is_dir() and p.name not in ('confs', 'union_check')):
        s = scores(d)
        if s:
            table.append((d.name, s, (d / 'DONE').exists()))
    table.sort(key=lambda t: -t[1]['best'])
    f = lambda v: f'{v:7.4f}' if v is not None else f"{'-':>7}"
    print(f"{'best':>7} {'bal':>7} {'last':>7} {'@rnd':>5} {'ceil':>7} {'fidel':>7} {'s.fit':>7} {'divers':>7}  run")
    for n, s, done in table:
        print(f"{s['best']:7.4f} {s['balanced']:7.4f} {s['last']:7.4f} {s['best_round']:5d} {f(s['ceiling'])} "
              f"{f(s['fidelity_mean'])} {f(s['synth_fit_mean'])} {f(s['diversity_mean'])}  {n}"
              f"{'' if done else '  (running)'}  {s['by_dataset']}")
    (Path(out) / 'summary.json').write_text(json.dumps([dict(run=n, done=d, **s) for n, s, d in table], indent=1))


def report(run):
    """Last round's --diagnostics of one run: per class, worst first."""
    rows = [json.loads(l) for l in (Path(run) / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    dg = rows[-1].get('diag')
    if not dg:
        return print(f'{run}: no diagnostics (run with --diagnostics)')
    print(f"round {rows[-1]['round']}: acc {rows[-1]['accuracy']:.4f}, real-data ceiling {dg['ceiling']:.4f}")
    print(f"{'test acc':>8} {'ceiling':>8}  class")
    for k, v in sorted(dg['per_class'].items(), key=lambda kv: kv[1]):
        print(f"{v:8.3f} {dg['ceiling_per_class'].get(k, float('nan')):8.3f}  {k}")
    print(f"\n{'fidelity':>8} {'syn.fit':>8} {'divers':>8}  generator row")
    for k in sorted(dg['synth_fit'], key=lambda k: dg['fidelity'].get(k, -1)):
        print(f"{dg['fidelity'].get(k, float('nan')):8.3f} {dg['synth_fit'][k]:8.3f} {dg['diversity'][k]:8.3f}  {k}")
    print('\ntop confusions (true -> predicted, count):')
    for t, pr, c in dg['confusions']:
        print(f'  {t} -> {pr}: {c}')
    print(f"\nsample grids: {Path(run) / 'diag'}/round_XXXX.png (8 per row, rows as above by index)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('runs/sweep'))
    p.add_argument('--scheme', choices=tuple(SCHEMES), default='plain')
    p.add_argument('--grid', help='JSON dict replacing the default grid, or a JSON list of explicit runs (dicts)')
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--jobs', type=int, default=3, help='runs at once')
    p.add_argument('--workers', type=int, default=1, help='client threads per run')
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpus', default='', help='e.g. "0 1": runs round-robin over these GPUs')
    p.add_argument('--data', default=DATA)
    p.add_argument('--extra', default='--keep-frac 1.0', help='flags for every run')
    p.add_argument('--conf', type=Path, default=Path('config.yaml'))
    p.add_argument('--min-mcc', type=float, default=1.0,
                   help='first build the label union alone (<out>/union_check), print its table and MCC, and start '
                        'the sweep only if MCC >= this (-1: always start)')
    p.add_argument('--report', type=Path, help='only print the per-class diagnostics of this run folder')
    p.add_argument('--summary', type=Path, help='only print the table of this sweep folder')
    a = p.parse_args()
    if a.report:
        return report(a.report)
    if a.summary:
        return summary(a.summary)
    grid = json.loads(a.grid) if a.grid else GRID
    combos = grid if isinstance(grid, list) else [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    gpus = a.gpus.split()
    (a.out / 'confs').mkdir(parents=True, exist_ok=True)
    print(f'{len(combos)} runs, {a.jobs} at a time -> {a.out}', flush=True)
    u = a.out / 'union_check'                         # union only (0 rounds): also loads / downloads the data
    if not (u / 'evaluator' / 'union.json').exists():
        u.mkdir(parents=True, exist_ok=True)
        print('union check ...', flush=True)
        with open(u / 'run.log', 'w') as log:
            if subprocess.call([sys.executable, '-m', 'secure_code_no_cluster', *a.data.split(), *SCHEMES[a.scheme],
                                *a.extra.split(), '--rounds', '0', '--device', a.device, '--no-progress',
                                '--output', str(u)], stdout=log, stderr=subprocess.STDOUT):
                sys.exit(f'union check failed: {u}/run.log')
    from secure_code_no_cluster import format_union
    ev = json.loads((u / 'evaluator' / 'union.json').read_text())
    m = ev['union_metrics']
    print(format_union(ev['experimenter_view']))
    print(f"union MCC={m['mcc']:.4f} exact={m['exact']} size={m['union_size']} (true {m['true_union_size']}) "
          f"split={m['split_labels']} merged={m['merged_indices']} missing={m['missing']} spurious={m['spurious']}",
          flush=True)
    if m['mcc'] < a.min_mcc:
        sys.exit(f"union MCC {m['mcc']:.4f} < --min-mcc {a.min_mcc}: sweep not started")

    loaded = threading.Event()        # later runs start once run 0 has loaded the data (args.json): datasets
                                      # are downloaded and splits written by one process only

    def one(i_combo):
        i, combo = i_combo
        name = name_of(combo)
        d = a.out / name
        if (d / 'DONE').exists():
            if i == 0:
                loaded.set()
            return
        if i:
            loaded.wait()
        conf = OmegaConf.load(a.conf)
        cli = []
        for k, v in combo.items():
            if k.startswith('--'):
                cli += [] if v is None else [k] if v == '' else [k, str(v)]
            else:
                conf[k] = v
        yml = a.out / 'confs' / f'{name}.yaml'
        OmegaConf.save(conf, yml)
        d.mkdir(exist_ok=True)
        resume = ['--resume', str(d / 'checkpoint_last.pt')] if (d / 'checkpoint_last.pt').exists() else []
        cmd = [sys.executable, '-m', 'secure_code_no_cluster', *a.data.split(), *SCHEMES[a.scheme],
               '--exp-conf', str(yml), *cli, *a.extra.split(), '--rounds', str(a.rounds), '--device', a.device,
               '--workers', str(a.workers), '--no-progress', '--output', str(d), *resume]
        env = dict(os.environ, **({'CUDA_VISIBLE_DEVICES': gpus[i % len(gpus)]} if gpus else {}))
        with open(d / 'run.log', 'a') as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            while i == 0 and not loaded.is_set() and p.poll() is None:
                if (d / 'args.json').exists():
                    loaded.set()
                time.sleep(5)
            if i == 0:
                loaded.set()                                       # failed early: let the others try
            rc = p.wait()
        if rc == 0:
            (d / 'DONE').touch()
        s = scores(d)
        print(f"{'done' if rc == 0 else f'FAIL rc={rc}'}  {name}  best={s['best'] if s else '-'}", flush=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(one, enumerate(combos)))
    summary(a.out)


if __name__ == '__main__':
    main()
