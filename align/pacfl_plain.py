"""P1: plaintext reference for private PACFL (what P2/P3 must reproduce under MPC).

Per client (local, nothing sent):
  1. project flattened images with a PUBLIC random matrix R (d -> r, e.g. 3072 -> 256);
     principal angles are approximately preserved (Johnson-Lindenstrauss)
  2. PACFL basis: per-label truncated SVD in the projected space, stacked and
     orthonormalised (QR) -> U_i (r x p_i)
Pairwise similarity (to be computed under OLE in P2):
  s_ij = ||U_i^T U_j||_F^2 / min(p_i, p_j) = <vec(P_i), vec(P_j)> / min(p_i, p_j),  P = U U^T
  in [0, 1]: sum of cos^2 of the principal angles, normalised.
Groups (to be computed in MPC in P3): connected components of the graph s_ij > tau
(single linkage). original_pacfl() runs the baseline (smallest angle + average linkage)
for comparison; agreement is measured with the adjusted Rand index.
"""
from collections import defaultdict

import numpy as np

PUBLIC_SEED = 20260926


def projection_matrix(in_dim, out_dim=256, seed=PUBLIC_SEED):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((in_dim, out_dim)) / np.sqrt(out_dim)).astype(np.float64)


def _collect(loader, samples_per_label):
    images, counts = defaultdict(list), defaultdict(int)
    for x, y in loader:
        x, y = np.asarray(x.detach().cpu().numpy() if hasattr(x, 'detach') else x), np.asarray(y)
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            counts[int(label)] += len(idx)
            have = sum(len(a) for a in images[int(label)])
            if have < samples_per_label:
                images[int(label)].append(x[idx[:samples_per_label - have]])
    if not counts:
        raise ValueError('PACFL requires nonempty client training data')
    return images, counts


def projected_basis(loader, R, budget=20, samples_per_label=64):
    """U_i: orthonormal r x p basis of the client's projected data (PACFL budget split by label frequency)."""
    images, counts = _collect(loader, samples_per_label)
    labels = sorted(counts)
    sizes = np.array([counts[l] for l in labels], float)
    raw = sizes / sizes.sum() * budget
    alloc = np.floor(raw).astype(int)
    alloc[np.argsort(raw - alloc)[::-1][:budget - alloc.sum()]] += 1
    cols = []
    for label, k in zip(labels, alloc):
        if k == 0:
            continue
        x = np.concatenate(images[label]).reshape(-1, R.shape[0]) * .5 + .5   # same affine as PACFL
        u, _, _ = np.linalg.svd((x @ R).T, full_matrices=False)
        cols.append(u[:, :min(k, u.shape[1])])
    q, _ = np.linalg.qr(np.hstack(cols))
    return q


def similarity(Ui, Uj):
    return float(np.linalg.norm(Ui.T @ Uj) ** 2 / min(Ui.shape[1], Uj.shape[1]))


def similarity_matrix(bases):
    n = len(bases)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = similarity(bases[i], bases[j])
    return S


def components(adjacency):
    """Connected components of a boolean adjacency matrix -> sorted list of sorted groups."""
    n, seen, groups = len(adjacency), set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, group = [s], []
        seen.add(s)
        while stack:
            v = stack.pop()
            group.append(v)
            for w in np.flatnonzero(adjacency[v]):
                if int(w) not in seen:
                    seen.add(int(w)); stack.append(int(w))
        groups.append(sorted(group))
    return sorted(groups)


def plain_groups(bases, tau):
    S = similarity_matrix(bases)
    return components(S > tau), S


def original_pacfl(raw_bases, thresh=20):
    from pacfl_utils import calculating_adjacency, hierarchical_clustering
    A = calculating_adjacency(list(range(len(raw_bases))), raw_bases)
    return sorted(sorted(g) for g in hierarchical_clustering(A, thresh=thresh, linkage='average'))


def labels_of(groups, n):
    out = np.zeros(n, int)
    for k, g in enumerate(groups):
        out[g] = k
    return out


def agreement(groups_a, groups_b, n):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_of(groups_a, n), labels_of(groups_b, n)))
