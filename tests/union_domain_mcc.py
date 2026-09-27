"""Label-union MCC with and without the weak image (domain) check, on real MNIST / EMNIST / CIFAR-10.

  python -m tests.union_domain_mcc [--clients-per-dataset 10] [--clash 5] [--samples 16]

Clients hold random label subsets of one dataset and use its real class names. --clash adds
clients whose names collide across domains: MNIST digit images named after CIFAR classes, and
CIFAR photos named '0'..'9'. Ground truth: a label is (name, kind of picture), so a digit called
'cat' is NOT the photo 'cat', while MNIST '3' and EMNIST '3' ARE the same label.
Scored per union entry with label_union.index_metrics over the universe name x kind
(TP/FP/FN/TN, MCC, split labels, merged indices).
"""
import argparse
import json

import numpy as np

from label_union import oprf_union, index_metrics
from label_union.domain import anchors, domain_code

KIND = {'MNIST': 'strokes', 'EMNIST': 'strokes', 'CIFAR10': 'photo'}


def load(name, root):
    from fl_datasets import get_raw_dataset_transform
    from setup import label_names
    ds = get_raw_dataset_transform(name, root)
    return ds, label_names(ds, name), np.asarray(ds.targets)


def build(args, rng):
    data = {d: load(d, args.data_root) for d in KIND}
    clients = []                                   # (dataset, {name: images}, rename)

    def client(d, classes, rename=None):
        ds, names, t = data[d]
        lab = {}
        for c in classes:
            idx = rng.choice(np.flatnonzero(t == c), args.samples, replace=False)
            lab[(rename or names)[c]] = np.stack([ds[int(i)][0].numpy() for i in idx])
        clients.append((d, lab))
    for d in KIND:
        n = len(data[d][1])
        for _ in range(args.clients_per_dataset):
            client(d, rng.choice(n, rng.integers(2, min(n, 20) + 1), replace=False))
    cifar_names = list(data['CIFAR10'][1])
    for _ in range(args.clash):
        client('MNIST', rng.choice(10, 5, replace=False), rename=cifar_names)            # digit "cat"
        client('CIFAR10', rng.choice(10, 5, replace=False), rename=[str(i) for i in range(10)])  # photo "3"
    return clients


def score(clients, index, U):
    concepts = [[f'{x}@{KIND[d]}' for x in lab] for d, lab in clients]
    by_concept = [{f'{x}@{KIND[d]}': idx[x] for x in lab} for (d, lab), idx in zip(clients, index)]
    universe = sorted({f'{x}@{k}' for _, lab in clients for x in lab for k in set(KIND.values())})   # name x kind
    m = index_metrics(concepts, by_concept, U, universe)
    return {k: m[k] for k in ('mcc', 'f1', 'exact', 'true_positive', 'false_positive', 'false_negative',
                              'true_negative', 'index_list_size', 'true_union_size', 'split_labels', 'merged_indices')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='data/raw')
    p.add_argument('--clients-per-dataset', type=int, default=10)
    p.add_argument('--clash', type=int, default=5, help='clash clients per direction (0: none)')
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    clients = build(args, np.random.default_rng(args.seed))
    names = [list(lab) for _, lab in clients]
    A = anchors(32)
    coded = [{x: domain_code(v, A) for x, v in lab.items()} for _, lab in clients]
    wrong = [(d, x, c) for (d, lab), cd in zip(clients, coded) for x, (c, _) in cd.items() if c != KIND[d]]
    out = dict(clients=len(clients), labels=sum(map(len, names)),
               domain_code_errors=len(wrong), min_margin=round(min(m for cd in coded for _, m in cd.values()), 3))
    for check in (False, True):
        domains = [{x: c for x, (c, _) in cd.items()} for cd in coded] if check else None
        index, U, stats = oprf_union(names, workers=args.workers, domains=domains)
        out['with_domain_check' if check else 'names_only'] = score(clients, index, U)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    main()
