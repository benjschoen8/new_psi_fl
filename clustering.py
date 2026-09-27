"""PACFL structural statistics and clustering, extracted from the baseline."""
from collections import defaultdict
import numpy as np
from pacfl_utils import calculating_adjacency, hierarchical_clustering


def local_basis(loader, budget=20, samples_per_label=64):
    if budget < 1 or samples_per_label < 1:
        raise ValueError('PACFL basis budget and sample cap must be positive')
    images, counts, stored = defaultdict(list), defaultdict(int), defaultdict(int)
    for x, y in loader:
        x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        for label in np.unique(y):
            label = int(label)
            indices = np.where(y == label)[0]
            counts[label] += len(indices)
            remaining = samples_per_label - stored[label]
            if remaining > 0:
                chosen = x[indices[:remaining]]
                images[label].append(chosen)
                stored[label] += len(chosen)
    if not counts:
        raise ValueError('PACFL requires nonempty client training data')
    labels = sorted(counts)
    sizes = np.array([counts[label] for label in labels], dtype=float)
    raw = sizes / sizes.sum() * budget
    allocated = np.floor(raw).astype(int)
    order = np.argsort(raw - allocated)[::-1]
    allocated[order[:budget - allocated.sum()]] += 1
    bases = []
    for label, count in zip(labels, allocated):
        if count == 0:
            continue
        samples = np.concatenate(images[label])
        matrix = samples.reshape(len(samples), -1).T * .5 + .5
        u, _, _ = np.linalg.svd(matrix, full_matrices=False)
        u = u / np.linalg.norm(u, ord=2, axis=0)
        bases.append(u[:, :min(count, u.shape[1])])
    return np.hstack(bases)


class PACFL:
    def __init__(self, threshold=20):
        if not 0 <= threshold <= 180:
            raise ValueError('PACFL angle threshold must be between 0 and 180')
        self.threshold = threshold

    def __call__(self, bases):
        ids = list(bases)
        if not ids:
            raise ValueError('Clustering requires clients')
        adjacency = calculating_adjacency(list(range(len(ids))), [bases[i] for i in ids])
        groups = hierarchical_clustering(adjacency, thresh=self.threshold, linkage='average')
        return {ids[index]: f'Cluster_{group}' for group, indices in enumerate(groups)
                for index in indices}

