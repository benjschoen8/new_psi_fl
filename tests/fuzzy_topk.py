"""Test of fuzzy "option B": every label enters its top-k anchors, and two labels are grouped when they
share at least r of them (connected components). k=1, r=1 is the current protocol (nearest anchor).
Run on a machine with the encoder (or its cache):

  python -m tests.fuzzy_topk                          # synonym writers (car / automobile / auto ...)
  python -m tests.fuzzy_topk --langs en,zh,es,ja,fr,de
  python -m tests.fuzzy_topk --langs en0,en1          # the current keywords (word / Capitalised word)

Grouping is done in the clear here (best case for option B). A secure version would have to reveal
which anchors belong to the same label to whoever groups, i.e. an anonymous similarity graph:
the leakage the single-anchor protocol avoids. Scores: pairwise MCC / precision / recall over all
(client, label) instances vs the true class, per dataset too.
"""
import argparse

import numpy as np

from label_union.encoder import DEFAULT_MODEL, embed
from label_union.fuzzy_union import anchor_words, components, false_friends, hub_penalty, normalize
from rt_descriptions import keyword
from tests.fuzzy_threshold import DATASETS, DOMAIN, score


def top_sets(E, texts, A, pen, skip, k, domain):
    """Per keyword: its set of top-k anchor keys under the CSLS score (single letters: their own text)."""
    sc = 2 * (E @ A.T) - pen
    if len(skip):
        sc[:, skip] = -np.inf
    top = np.argsort(-sc, 1)[:, :k]
    out = []
    for t, row in zip(texts, top):
        keys = {f'text:{t}'} if len(t) == 1 and t.isascii() and t.isalpha() else {f'anchor:{j}' for j in row}
        out.append({f'{key}|{domain}' for key in keys})
    return out


def group(sets, r):
    """Link two instances sharing >= min(r, |S_i|, |S_j|) keys; connected components."""
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            need = min(r, len(sets[i]), len(sets[j]))
            if len(sets[i] & sets[j]) >= need:
                pairs.append((i, j))
    return components(len(sets), pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--langs', default='syn0,syn1,syn2')
    ap.add_argument('--anchors', default='2000,5000,20000')
    ap.add_argument('--hubs', default='0,5')
    ap.add_argument('--topk', default='1,2,3,5')
    ap.add_argument('--share', default='1,2,3', help='min shared anchors to link two labels (capped at k)')
    a = ap.parse_args()
    langs = a.langs.split(',')
    Ns = sorted(int(x) for x in a.anchors.split(','))
    A_all = embed(anchor_words(Ns[-1]), a.model)
    clients = [(d, lang, DATASETS[d]) for d in DATASETS for lang in langs]
    texts = [[normalize(keyword(d, x, lang)) for x in labels] for d, lang, labels in clients]
    E = [embed(t, a.model) for t in texts]
    truth = np.array([x for _, _, labels in clients for x in labels])
    ds_of = np.array([d for d, _, labels in clients for _ in labels])
    print(f'{len(truth)} labels ({len(set(truth))} true classes), writers {langs}\n')
    print(f"{'N':>6} {'hub':>3} {'k':>2} {'r':>2}   {'MCC':>6} {'P':>6} {'R':>6} {'groups':>6}   per dataset MCC")
    rows = []
    for N in Ns:
        A = A_all[:N]
        for hub in map(int, a.hubs.split(',')):
            pen = hub_penalty(A, hub)
            for k in map(int, a.topk.split(',')):
                sets = [s for (d, lang, _), t, e in zip(clients, texts, E)
                        for s in top_sets(e, t, A, pen, false_friends(lang, N), k, DOMAIN[d])]
                for r in sorted({min(int(x), k) for x in a.share.split(',')}):
                    g = group(sets, r)
                    row = dict(N=N, hub=hub, k=k, r=r, groups=len(set(g)), **score(g, truth),
                               per={d: score(g, truth, ds_of == d)['mcc'] for d in DATASETS})
                    rows.append(row)
                    print(f"{N:>6} {hub:>3} {k:>2} {r:>2}   {row['mcc']:.3f} {row['precision']:.3f} {row['recall']:.3f} "
                          f"{row['groups']:>6}   " + ' '.join(f'{d}={v:.2f}' for d, v in row['per'].items()), flush=True)
    base = max((r for r in rows if r['k'] == 1), key=lambda r: r['mcc'])
    best = max(rows, key=lambda r: r['mcc'])
    print(f"\nsingle anchor (k=1, current protocol): best MCC {base['mcc']:.3f} at N={base['N']} hub={base['hub']}")
    print(f"option B best: MCC {best['mcc']:.3f} at N={best['N']} hub={best['hub']} k={best['k']} r={best['r']} "
          f"(P={best['precision']:.3f} R={best['recall']:.3f}, {best['groups']} groups)")


if __name__ == '__main__':
    main()
