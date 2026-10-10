"""Plaintext sweep of the similar-union thresholds on a real setup split (experimenter tool, no MPC):
keyword CSLS threshold tau, image anchors per label k and required overlap t. For every setting: union
size U, pair MCC against the true names, propagation steps, wrong merges / splits. The grouping is the
ideal functionality of the MPC (same rows, PCA, fixed point), so the MPC gives the same result.

  python -m tests.threshold_sweep --datasets MNIST,EMNIST,CIFAR10,FashionMNIST,STL10,CIFAR100,USPS --clients 30
  python -m tests.threshold_sweep ... --taus 0.10 0.15 0.20 --ks 6 8 --show 0.15,8,3    # list errors of one
"""
import argparse
import itertools
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def load(a):
    from setup_smoke import real_inputs
    args = SimpleNamespace(datasets=a.datasets, num_train_cifar10stl10=0, class_subsets='100,100', class_share='full',
                           noniid_partition='dirichlet', seed=a.seed, data_root=a.data_root, exp_conf=a.exp_conf,
                           fuzzy_langs='en0,en1', samples_per_label=16)
    labels, samples, keywords, _ = real_inputs(a.clients, args)
    return labels, samples, keywords


def rows(labels, samples, keywords, pca, ks):
    from label_union import circuit_union as cu
    from label_union.domain import client_sets
    from label_union.pca import project
    kws = [cu.keyword_rows(own, True) for own in keywords]
    kws = [project(own, pca) for own in kws] if pca else kws
    owner, name, kind, emb, r, sym = [], [], [], [], [], []
    for c, own in enumerate(labels):
        for x in own:
            k = kws[c][x]
            owner.append(c), name.append(x), kind.append(k[0])
            emb.append(np.asarray(k[1], np.int64) if k[0] == 'emb' else None)
            r.append(int(k[2]) if k[0] == 'emb' else 0)
            sym.append(k[1] if k[0] == 'sym' else None)
    img = {}
    for k in ks:                                                  # anchor bit vectors per label, k anchors each
        vec = []
        for s, own in zip(samples, labels):
            v = cu.image_rows(client_sets(s, own, k))
            vec += [np.asarray(v[x], np.int64) for x in own]
        img[k] = np.stack(vec)
    return dict(owner=np.array(owner), name=np.array(name, object), kind=np.array(kind), emb=emb, r=np.array(r),
                sym=sym, img=img)


def kw_scores(R):
    """CSLS score (fixed point 2^14) for embedding pairs, +inf for equal symbols, -inf otherwise."""
    N = len(R['owner'])
    S = np.full((N, N), -np.inf)
    e = [i for i in range(N) if R['kind'][i] == 'emb']
    E = np.stack([R['emb'][i] for i in e])
    S[np.ix_(e, e)] = 2 * (E @ E.T) - R['r'][e][:, None] - R['r'][e][None, :]
    s = [i for i in range(N) if R['kind'][i] == 'sym']
    for i, j in itertools.product(s, s):
        if R['sym'][i] == R['sym'][j]:
            S[i, j] = np.inf
    return S


def evaluate(R, S, tau, k, t):
    from label_union.circuit_union import FIX
    diff = R['owner'][:, None] != R['owner'][None, :]
    ov = R['img'][k] @ R['img'][k].T
    A = diff & (S >= round(tau * (1 << 2 * FIX))) & (ov >= t)
    n, comp = connected_components(csr_matrix(A), directed=False)
    same = R['name'][:, None] == R['name'][None, :]
    pred = comp[:, None] == comp[None, :]
    iu = np.triu_indices(len(comp), 1)
    m = diff[iu]
    sa, pr = same[iu][m], pred[iu][m]
    tp, fp, fn = int((sa & pr).sum()), int((~sa & pr).sum()), int((sa & ~pr).sum())
    tn = int(m.sum()) - tp - fp - fn
    den = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / den ** .5 if den else float(fp == fn == 0)
    steps = 0                                                     # eccentricity of each component's minimum
    D = shortest_path(csr_matrix(A), unweighted=True, directed=False,
                      indices=[int(np.flatnonzero(comp == c)[0]) for c in range(n)])
    for c in range(n):
        d = D[c][comp == c]
        steps = max(steps, int(d[np.isfinite(d)].max()))
    merged = [sorted(set(R['name'][comp == c])) for c in range(n) if len(set(R['name'][comp == c])) > 1]
    split = sorted((x, len(set(comp[R['name'] == x]))) for x in set(R['name']) if len(set(comp[R['name'] == x])) > 1)
    return dict(tau=tau, k=k, t=t, U=n, true_U=len(set(R['name'])), mcc=mcc, steps=steps, fp=fp, fn=fn,
                merged=merged, split=split)


def main(argv=None):
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', default='MNIST,EMNIST,CIFAR10,FashionMNIST,STL10,CIFAR100,USPS')
    p.add_argument('--clients', type=int, default=30)
    p.add_argument('--pca', type=int, default=48)
    p.add_argument('--taus', type=float, nargs='+', default=[.10, .12, .15, .18, .20, .25, .30])
    p.add_argument('--ks', type=int, nargs='+', default=[6, 8, 10])
    p.add_argument('--show', default=None, help='tau,k,t: print the wrong merges / splits of this setting')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--data-root', type=Path, default=Path('data/raw'))
    p.add_argument('--exp-conf', type=Path, default=Path('config.yaml'))
    a = p.parse_args(argv)
    R = rows(*load(a), a.pca, a.ks)
    S = kw_scores(R)
    res = [evaluate(R, S, tau, k, t) for tau in a.taus for k in a.ks for t in range(0, k + 1)]
    print(f"{'tau':>5} {'k':>3} {'t':>3} {'U':>5} {'true':>5} {'MCC':>7} {'steps':>6} {'wrong merges':>13} {'splits':>7}")
    for x in sorted(res, key=lambda x: (-x['mcc'], x['steps']))[:40]:
        print(f"{x['tau']:5.2f} {x['k']:3d} {x['t']:3d} {x['U']:5d} {x['true_U']:5d} {x['mcc']:7.4f} {x['steps']:6d} "
              f"{len(x['merged']):13d} {len(x['split']):7d}")
    cur = next(x for x in res if abs(x['tau'] - .10) < 1e-9 and x['k'] == 6 and x['t'] == 2) if .10 in a.taus and 6 in a.ks else None
    if cur:
        print(f"\ncurrent setting (tau .10, k 6, t 2): MCC {cur['mcc']:.4f}, U {cur['U']}/{cur['true_U']}, steps {cur['steps']}")
    if a.show:
        tau, k, t = a.show.split(',')
        x = next(x for x in res if abs(x['tau'] - float(tau)) < 1e-9 and x['k'] == int(k) and x['t'] == int(t))
        print(f"\n{a.show}: MCC {x['mcc']:.4f}, U {x['U']}/{x['true_U']}, steps {x['steps']}")
        for g in x['merged']:
            print('  merged:', g)
        print('  split:', x['split'])


if __name__ == '__main__':
    main()
