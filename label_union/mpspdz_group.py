"""Step C of circuit_union (match + connected components + K to owner) as a real MPC in MP-SPDZ:
Shamir secret sharing, honest majority (shamir-party.x, threshold < n/2), n parties on this host.

    MPSPDZ=/path/to/mp-spdz python -m label_union.mpspdz_group            # self-check + one benchmark
    MPSPDZ=... python -m label_union.mpspdz_group --n 3 5 --m 10 --fuzzy  # benchmark grid

mpspdz_group(rows, owners, ...) takes the rows of circuit_union.group() and returns the same root per
row (smallest row of the component, in group()'s numbering), the measured cost and every row's K.
circuit_union_with_keys runs step C here whenever MPSPDZ is set (else as its ideal functionality). The circuit is
mpc/circuit_group.mpc. Keyword names and symbols enter as 54-bit hashes (collision 2^-54 per pair).
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / 'mpc' / 'circuit_group.mpc'
HB = 54


def _h(b) -> int:
    b = b if isinstance(b, bytes) else str(b).encode()
    return int.from_bytes(hashlib.sha256(b'kw/' + b).digest()[:8], 'big') >> (64 - HB)


def _home(root=None) -> Path:
    root = Path(root or os.environ.get('MPSPDZ', ''))
    if not (root / 'compile.py').exists() or not (root / 'shamir-party.x').exists():
        raise SystemExit('set MPSPDZ to an MP-SPDZ directory with compile.py and shamir-party.x '
                         '(binary release: run Scripts/tldr.sh and Scripts/setup-ssl.sh <n> once)')
    return root


_runs = itertools.count()


def _free_ports(n):
    """Base port with n free consecutive ports (party p listens on base + p)."""
    for _ in range(100):
        base = random.randrange(20000, 60000)
        try:
            socks = [socket.create_server(('localhost', base + p)) for p in range(n)]
        except OSError:
            continue
        for s in socks:
            s.close()
        return base
    raise RuntimeError('no free ports')


def _row_tokens(kw, img, fuzzy, sym, d, nimg):
    if kw is None:                                                    # dummy row
        out = [0] + ([0, 0] if sym else []) + ([0] * (d + 1) if fuzzy else [0])
        return out + [0] * nimg
    out = [1]
    if not fuzzy:
        out.append(_h(kw[1]))
    else:
        if sym:
            out += [1, _h(kw[1])] if kw[0] == 'sym' else [0, 0]
        out += ([0] * (d + 1) if kw[0] == 'sym' else [int(x) for x in kw[1]] + [int(kw[2])])
    return out + ([int(x) for x in img] if nimg else [])


def mpspdz_group(rows, owners, tau=0.10, t=2, m=None, root=None, port=None, fix=7, timeout=None, edabit=True):
    """rows: list of (kw, img or None) as in circuit_union.group(); owners: client of each row.
    Returns (root per row, stats, K per row as int: the opened kappa of the row's root, each value
    seen only by the row's owner). Rows of a client are padded with dummies to m (default: max)."""
    home = _home(root)
    n = max(owners) + 1
    if n < 3:
        raise ValueError('honest-majority Shamir needs n >= 3 parties')
    per = [[a for a, c in enumerate(owners) if c == p] for p in range(n)]
    m = m or max(map(len, per))
    kinds = {kw[0] for kw, _ in rows}
    fuzzy = 'name' not in kinds
    sym = int(fuzzy and 'sym' in kinds)
    embs = [kw[1] for kw, _ in rows if kw[0] == 'emb']
    d = len(embs[0]) if embs else 1
    nimg = 0 if rows[0][1] is None else len(rows[0][1])
    tau_i = int(round(tau * (1 << 2 * fix)))
    args = [f'n={n}', f'm={m}', f'd={d if fuzzy else 0}', f'nimg={nimg}', f'tau={tau_i}', f't={t}',
            f"mode={'fuzzy' if fuzzy else 'exact'}", f'sym={sym}']

    shutil.copy(SRC, home / 'Programs' / 'Source' / SRC.name)
    t0 = time.perf_counter()
    out = subprocess.run(['python3', 'compile.py', *(['-Y'] if edabit else []), SRC.stem, *args], cwd=home, capture_output=True,
                         text=True, check=True).stdout
    name = re.search(r'Writing to .*?Programs/Schedules/(\S+)\.sch', out).group(1)
    compile_s = time.perf_counter() - t0

    tag = f'cg{os.getpid()}-{next(_runs)}'                # own input/output files: runs may overlap
    port = port or _free_ports(n)
    (home / 'Player-Data').mkdir(exist_ok=True)
    for p in range(n):
        toks = []
        for i in range(m):
            a = per[p][i] if i < len(per[p]) else None
            kw, img = rows[a] if a is not None else (None, None)
            toks += _row_tokens(kw, img, fuzzy, sym, d, nimg)
        (home / 'Player-Data' / f'{tag}-Input-P{p}-0').write_text(' '.join(map(str, toks)) + '\n')

    t0 = time.perf_counter()
    procs = [subprocess.Popen(['./shamir-party.x', '-N', str(n), '-p', str(p), '-pn', str(port),
                               '-h', 'localhost', '-IF', f'Player-Data/{tag}-Input', '-OF', tag, name], cwd=home,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for p in range(n)]
    logs = [pr.communicate(timeout=timeout)[0] for pr in procs]
    run_s = time.perf_counter() - t0
    for p, pr in enumerate(procs):
        if pr.returncode:
            raise RuntimeError(f'party {p} failed:\n{logs[p][-3000:]}')

    K = {}
    for p in range(n):
        text = (home / f'{tag}-P{p}-0').read_text()
        for i, k in re.findall(r'^K (\d+) (-?\d+)$', text, re.M):
            if int(i) < len(per[p]):
                K[per[p][int(i)]] = int(k)
    first = {}
    root_ = []
    for a in range(len(rows)):                        # root = smallest row with the same K
        root_.append(first.setdefault(K[a], a))

    log0 = logs[0] + (home / f'{tag}-P0-0').read_text()
    for f in [*home.glob(f'{tag}-P*'), *(home / 'Player-Data').glob(f'{tag}-Input-*')]:
        f.unlink()
    num = lambda pat: float(re.search(pat, log0).group(1)) if re.search(pat, log0) else None
    stats = dict(edabit=edabit, n=n, m=m, rows=n * m, d=d if fuzzy else 0, nimg=nimg, mode='fuzzy' if fuzzy else 'exact',
                 sym=sym, compile_seconds=round(compile_s, 2), wall_seconds=round(run_s, 2),
                 time_seconds=num(r'Time = ([\d.e+-]+) seconds'),
                 party0_MB=num(r'Data sent = ([\d.e+-]+) MB'),
                 rounds=num(r'in ~(\d+) rounds'),
                 global_MB=num(r'Global data sent = ([\d.e+-]+) MB'),
                 propagation_steps=num(r'propagation steps \(incl\. the public first step\): (\d+)'))
    return root_, stats, [K[a] for a in range(len(rows))]


def _synthetic(n, m, n_groups, fuzzy, rng, d=384, nimg=45, k=6, real_frac=0.8):
    """Planted classes: every client holds about real_frac*m labels drawn from n_groups classes.
    Keywords: same class -> same name (exact) or a near-identical embedding; images: k anchors of
    the class's 'kind'."""
    from label_union.circuit_union import FIX
    base = rng.standard_normal((n_groups, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    kinds = [rng.choice(nimg, k, replace=False) for _ in range(n_groups)]
    rows, owners = [], []
    for c in range(n):
        for g in rng.choice(n_groups, int(real_frac * m), replace=False):
            if fuzzy:
                e = base[g] + 0.05 * rng.standard_normal(d)
                e /= np.linalg.norm(e)
                kw = ('emb', np.round(e * (1 << FIX)).astype(np.int64), int(round(0.3 * (1 << 2 * FIX))))
            else:
                kw = ('name', f'class{g}'.encode())
            img = np.zeros(nimg, np.int64)
            img[kinds[g]] = 1
            rows.append((kw, img))
            owners.append(c)
    return rows, owners


def main():
    from label_union.circuit_union import group, mpc_cost, TAU
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[3])
    ap.add_argument('--m', type=int, nargs='+', default=[10])
    ap.add_argument('--fuzzy', action='store_true')
    ap.add_argument('--groups', type=int, default=None, help='planted classes (default 1.5 m)')
    ap.add_argument('--out', default=None, help='append results as JSON lines')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-edabit', dest='edabit', action='store_false',
                    help='comparisons by plain bit decomposition (default: edaBits, compile -Y; ~2.5x less traffic)')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for n in a.n:
        for m in a.m:
            rows, owners = _synthetic(n, m, a.groups or int(1.5 * m), a.fuzzy, rng)
            ideal, steps = group(rows, TAU, 2, owners)
            got, st, _ = mpspdz_group(rows, owners, TAU, 2, m=m, edabit=a.edabit)
            ok = got == ideal
            st.update(groups=len(set(ideal)), equal_to_ideal=ok,
                      estimate=mpc_cost(n, m, st['d'] or 1, st['nimg'], 1, steps))
            print(json.dumps(st))
            if a.out:
                with open(a.out, 'a') as f:
                    f.write(json.dumps(st) + '\n')
            assert ok, 'MP-SPDZ grouping differs from the ideal functionality'


if __name__ == '__main__':
    main()
