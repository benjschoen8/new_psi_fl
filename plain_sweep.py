"""Plain-GeFL (--agg plain: no SecAgg, no PSI cryptography) hyper-parameter grid search for max accuracy.

    python plain_sweep.py                                   # default grid, 3-dataset clients, 15 rounds
    python plain_sweep.py --jobs 3 --rounds 20 --grid '{"--gen-widths": ["128,64,32"], "gen_lr": [2e-4, 5e-4]}'
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

GRID = {                                     # 3 x 2 x 2 x 2 = 24 runs
    '--gen-widths': ['64,32,16', '128,64,32', '256,128,64'],
    '--warmup-epochs': [0, 10],
    'global_samples_per_class': [64, 512],
    'global_model_epochs': [5, 20],
}
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
    return dict(best=best['accuracy'], best_round=best['round'], last=rows[-1]['accuracy'],
                rounds=rows[-1]['round'], balanced=bal, by_dataset=per)


def summary(out):
    table = []
    for d in sorted(p for p in Path(out).iterdir() if p.is_dir() and p.name != 'confs'):
        s = scores(d)
        if s:
            table.append((d.name, s, (d / 'DONE').exists()))
    table.sort(key=lambda t: -t[1]['best'])
    print(f"{'best':>7} {'bal':>7} {'last':>7} {'@rnd':>5}  run")
    for n, s, done in table:
        print(f"{s['best']:7.4f} {s['balanced']:7.4f} {s['last']:7.4f} {s['best_round']:5d}  {n}"
              f"{'' if done else '  (running)'}  {s['by_dataset']}")
    (Path(out) / 'summary.json').write_text(json.dumps([dict(run=n, done=d, **s) for n, s, d in table], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('runs/sweep'))
    p.add_argument('--grid', help='JSON dict replacing the default grid')
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--jobs', type=int, default=3, help='runs at once')
    p.add_argument('--workers', type=int, default=1, help='client threads per run')
    p.add_argument('--device', default='cuda')
    p.add_argument('--gpus', default='', help='e.g. "0 1": runs round-robin over these GPUs')
    p.add_argument('--data', default=DATA)
    p.add_argument('--extra', default='--keep-frac 1.0', help='flags for every run')
    p.add_argument('--conf', type=Path, default=Path('config.yaml'))
    p.add_argument('--summary', type=Path, help='only print the table of this sweep folder')
    a = p.parse_args()
    if a.summary:
        return summary(a.summary)
    grid = json.loads(a.grid) if a.grid else GRID
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    gpus = a.gpus.split()
    (a.out / 'confs').mkdir(parents=True, exist_ok=True)
    print(f'{len(combos)} runs, {a.jobs} at a time -> {a.out}', flush=True)

    def one(i_combo):
        i, combo = i_combo
        name = name_of(combo)
        d = a.out / name
        if (d / 'DONE').exists():
            return
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
        cmd = [sys.executable, '-m', 'secure_code_no_cluster', *a.data.split(), '--agg', 'plain',
               '--exp-conf', str(yml), *cli, *a.extra.split(), '--rounds', str(a.rounds), '--device', a.device,
               '--workers', str(a.workers), '--no-progress', '--output', str(d), *resume]
        env = dict(os.environ, **({'CUDA_VISIBLE_DEVICES': gpus[i % len(gpus)]} if gpus else {}))
        with open(d / 'run.log', 'a') as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            (d / 'DONE').touch()
        s = scores(d)
        print(f"{'done' if rc == 0 else f'FAIL rc={rc}'}  {name}  best={s['best'] if s else '-'}", flush=True)

    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(one, enumerate(combos)))
    summary(a.out)


if __name__ == '__main__':
    main()
