"""Choose the fuzzy-union parameters (label_union.fuzzy_union) with the real encoder. Run once on a
machine with internet:

  pip install sentence-transformers
  python -m tests.fuzzy_threshold          # writes label_union/fuzzy_params.json,
                                           # data/encoder/<model>.npz (anchor + keyword embeddings),
                                           # runs/fuzzy_threshold/{report.json, threshold.png}
Data: every class of MNIST / EMNIST / CIFAR-10 as the bare keyword a client would type
(rt_descriptions.keyword); one simulated client per (dataset, writer). Default: 2 English writers
('car' / 'Car', at most two keywords per class; letters as is); --langs en,zh,es,ja,fr,de for 6 languages.
Score: the protocol's grouping (snap to anchor class, exact union; in the clear, the secure version
gives the same groups), pairwise MCC over (client, label) instances vs the true class.
Grid: anchors N (vocabulary prefix) x merge (mutual-NN synonym cosine; 1 = none) x hub (CSLS k; 0 = plain
cosine) x floor (min snap cosine). Every client skips its language's false friends.
The log ends with every split class and every merged group of the best setting.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from label_union.encoder import DEFAULT_MODEL, embed
from label_union.fuzzy_union import (PARAMS_FILE, anchor_classes, anchor_words, client_keys, false_friends,
                                     hub_penalty, normalize, union)
from rt_descriptions import LANGS, keyword

DATASETS = {'MNIST': [str(d) for d in range(10)],
            'EMNIST': [str(d) for d in range(10)] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)],
            'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']}
DOMAIN = {'MNIST': 'strokes', 'EMNIST': 'strokes', 'CIFAR10': 'photo'}


def mcc(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.


def score(g, truth, mask=None):
    same_p, same_t = g[:, None] == g[None, :], truth[:, None] == truth[None, :]
    iu = np.triu_indices(len(g), 1)
    keep = np.ones(len(iu[0]), bool) if mask is None else (mask[iu[0]] & mask[iu[1]])
    p, t = same_p[iu][keep], same_t[iu][keep]
    tp, fp, fn, tn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())
    return dict(mcc=mcc(tp, fp, fn, tn), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--langs', default='en0,en1',
                    help=f'en0,en1 = English writers (word / Capitalised word); or {",".join(LANGS)}')
    ap.add_argument('--anchors', default='2000,3000,5000,10000,20000')
    ap.add_argument('--merges', default='1,.9,.7,.5')
    ap.add_argument('--hubs', default='0,5,10,20')
    ap.add_argument('--floors', default='0')
    ap.add_argument('--no-domain', action='store_true', help='ignore the image check')
    ap.add_argument('--out', type=Path, default=Path('runs/fuzzy_threshold'))
    ap.add_argument('--no-write-params', action='store_true')
    a = ap.parse_args()
    langs = a.langs.split(',')
    Ns = sorted(int(x) for x in a.anchors.split(','))
    words = anchor_words(Ns[-1])
    A = embed(words, a.model)                                            # also fills the cache
    clients = [(d, lang, DATASETS[d]) for d in DATASETS for lang in langs]
    texts = [[normalize(keyword(d, x, lang)) for x in labels] for d, lang, labels in clients]
    E = [embed(t, a.model) for t in texts]
    doms = [None if a.no_domain else {x: DOMAIN[d] for x in labels} for d, _, labels in clients]
    inst = [(i, x) for i, (_, _, labels) in enumerate(clients) for x in labels]
    truth = np.array([x for _, x in inst])
    ds_of = np.array([clients[i][0] for i, _ in inst])

    grid, pens = [], {}
    for merge in sorted(map(float, a.merges.split(',')), reverse=True):
        for N in Ns:
            cls = anchor_classes(A[:N], merge)
            for hub in map(int, a.hubs.split(',')):
                pen = pens.setdefault((N, hub), hub_penalty(A[:N], hub))
                for floor in map(float, a.floors.split(',')):
                    keys = [client_keys(labels, t, e, A[:N], cls, floor, dm, pen, false_friends(lang, N))
                            for (_, lang, labels), t, e, dm in zip(clients, texts, E, doms)]
                    index, _, _, U, _ = union(keys, secure=False)
                    g = np.array([index[i][x] for i, x in inst])
                    r = dict(anchors=N, merge=merge, hub=hub, floor=floor, groups=U, **score(g, truth),
                             per_dataset={d: score(g, truth, ds_of == d)['mcc'] for d in DATASETS})
                    grid.append(r)
                    print(f"N={N:>5} merge={merge:.2f} hub={hub:>2} floor={floor:.2f}  MCC={r['mcc']:.3f} P={r['precision']:.3f} "
                      f"R={r['recall']:.3f} groups={U} (true {len(set(truth))})  "
                          + ' '.join(f'{d}={v:.2f}' for d, v in r['per_dataset'].items()), flush=True)
    top = max(r['mcc'] for r in grid)                                    # near-ties: least merging, fewer anchors
    best = max((r for r in grid if r['mcc'] >= top - .005), key=lambda r: (r['merge'], -r['hub'], -r['anchors'], r['floor']))

    # what every keyword snapped to under the best setting (for the paper / sanity)
    N = best['anchors']
    cls, pen = anchor_classes(A[:N], best['merge']), pens[N, best['hub']]
    snaps, members = {}, {}
    keys = [client_keys(labels, t, e, A[:N], cls, best['floor'], dm, pen, false_friends(lang, N))
            for (_, lang, labels), t, e, dm in zip(clients, texts, E, doms)]
    for (d, lang, labels), t, e, k in zip(clients, texts, E, keys):
        S = e @ A[:N].T
        sc = 2 * S - pen
        sc[:, false_friends(lang, N)] = -np.inf
        j = sc.argmax(1)
        for x, w, jj, s in zip(labels, t, j, S[np.arange(len(j)), j]):
            snaps.setdefault(f'{d}/{x}', {})[lang] = f'{w} -> {words[jj]} (cos {s:.2f}) = {k[x]}'
            members.setdefault(k[x], set()).add(x)
    split = {c: v for c, v in snaps.items() if len({s.split(' = ')[1] for s in v.values()}) > 1}
    merged = {k: sorted(v) for k, v in members.items() if len(v) > 1}
    report = dict(model=a.model, langs=langs, domain_check=not a.no_domain, best=best, true_labels=len(set(truth)),
                  grid=grid, split=split, merged=merged, snaps=snaps)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not a.no_write_params:
        PARAMS_FILE.write_text(json.dumps(dict(model=a.model, anchors=best['anchors'], merge=best['merge'],
                                               hub=best['hub'], floor=best['floor']), indent=2))
    plot(grid, best, a.out / 'threshold.png')
    print(f'\n{len(split)} classes split across languages (best setting):')
    for c, v in split.items():
        print(f'  {c}: ' + ' | '.join(f'{l}: {s}' for l, s in v.items()))
    print(f'{len(merged)} groups holding several classes:')
    for k, v in merged.items():
        print(f'  {k}: {v}')
    print(f"\nbest: N={best['anchors']} merge={best['merge']} hub={best['hub']} floor={best['floor']}  MCC={best['mcc']:.3f} "
          f"P={best['precision']:.3f} R={best['recall']:.3f} groups={best['groups']}/{len(set(truth))}  "
          + ' '.join(f'{d}={v:.2f}' for d, v in best['per_dataset'].items()))
    print(f"wrote {a.out / 'report.json'} (see 'snaps' for every keyword -> anchor), {a.out / 'threshold.png'}"
          + ('' if a.no_write_params else f', {PARAMS_FILE}') + ', encoder cache in data/encoder/')


def plot(grid, best, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, grid_c, bg = '#52514e', '#e4e3df', '#fcfcfb'
    colors = ['#2a78d6', '#eb6834', '#1f9d6a', '#8a5cd6']
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=bg)
    for x in ax:
        x.set_facecolor(bg); x.grid(color=grid_c); x.set_axisbelow(True)
        for s in ('top', 'right'):
            x.spines[s].set_visible(False)
    for c, N in zip(colors, sorted({r['anchors'] for r in grid})):
        rs = sorted((r for r in grid if r['anchors'] == N and r['floor'] == best['floor'] and r['hub'] == best['hub']), key=lambda r: r['merge'])
        ax[0].plot([r['merge'] for r in rs], [r['mcc'] for r in rs], color=c, lw=2, marker='o', label=f'{N} anchors')
    ax[0].set(title=f"(a) Grouping quality (hub={best['hub']}, floor={best['floor']})", xlabel='synonym merge cosine (1 = none)',
              ylabel='MCC', ylim=(0, 1.02))
    ax[0].legend(frameon=False, loc='lower left')
    ds = list(best['per_dataset'])
    ax[1].bar(ds, [best['per_dataset'][d] for d in ds], color=colors[0])
    ax[1].set(title=f"(b) Per dataset at best (N={best['anchors']}, merge={best['merge']})", ylabel='MCC', ylim=(0, 1.02))
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=bg)


if __name__ == '__main__':
    main()
