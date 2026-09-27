"""Progress bars for every run of run_experiments.sh (reads the runs' files; never touches them).
States: loading (reading datasets, building clients) -> setup (label union, round 1) -> running -> done;
stalled? = no new output for 3 rounds + 10 min.

  python -m tests.progress                    # once
  python -m tests.progress --watch 30         # refresh every 30 s (Ctrl-C to quit)
  python -m tests.progress --out runs/paper
"""
import argparse
import json
import time
from pathlib import Path


def status(run, width):
    args = json.loads((run / 'args.json').read_text()) if (run / 'args.json').exists() else {}
    target = run / 'target_rounds'                                    # written by run_experiments.sh
    total = (int(target.read_text()) if target.exists() else
             int(args['rounds']) if str(args.get('rounds', 'None')).isdigit() else 45)
    rows = []
    if (run / 'metrics.jsonl').exists():
        for line in (run / 'metrics.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    done_rounds = rows[-1]['round'] if rows else 0
    total = max(total, done_rounds)                                   # resumed with more rounds
    per = sum(sum(r['seconds'].values()) for r in rows) / len(rows) if rows else None
    log = run.parent / 'logs' / f'{run.name}.log'
    if (run / 'DONE').exists():
        state, eta = 'done', ''
    else:
        idle = time.time() - max(p.stat().st_mtime for p in (log, run / 'metrics.jsonl') if p.exists()) \
            if log.exists() else None
        stalled = per is not None and idle is not None and idle > 3 * per + 600
        started = (run / 'args.json').exists()
        state = 'stalled?' if stalled else ('running' if rows else 'setup' if started else 'loading')
        eta = f'ETA {fmt((total - done_rounds) * per)}' if per else ''
    fill = int(width * done_rounds / total) if total else 0
    acc = f"acc {rows[-1]['accuracy']:.4f}" if rows else ''
    return f"{run.name:<15} [{'#' * fill}{'.' * (width - fill)}] {done_rounds:>3}/{total:<3} {state:<9} {acc:<11} {eta}"


def fmt(s):
    s = int(s)
    return f'{s // 3600}h{s % 3600 // 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


def show(out, width=30):
    out = Path(out)
    runs = sorted(p for p in out.iterdir() if p.is_dir() and
                  ((p / 'args.json').exists() or (p / 'target_rounds').exists())) if out.exists() else []
    lines = [time.strftime('%F %T') + f'   {out}'] + [status(r, width) for r in runs]
    if not runs:
        lines.append('(no runs started yet)')
    prog = Path(out) / 'progress.log'
    if prog.exists():
        events = [l for l in prog.read_text().splitlines() if l.startswith('[')]
        lines += ['', 'last events:'] + ['  ' + l for l in events[-4:]]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='runs/paper')
    p.add_argument('--watch', type=float, help='refresh every N seconds')
    a = p.parse_args()
    while True:
        text = show(a.out)
        print('\033[2J\033[H' + text if a.watch else text, flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    main()
