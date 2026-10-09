"""Public PCA of keyword embeddings (option pca_dim of the hybrid setup; any pair version).

The basis is fitted on the public CSLS anchor words only (never on client labels), so it is a
public parameter. Each client projects its own embeddings, renormalises them and recomputes its
CSLS hub term r against the projected anchors; the MPC then sees k instead of 384 coordinates
(hegc: the HE matrix product shrinks about with k). CSLS on k dimensions approximates CSLS on 384:
compare groupings with setup_smoke_hybrid --approx-check.
"""
import functools

import numpy as np

from label_union.circuit_union import FIX, HUB


@functools.lru_cache(maxsize=4)
def _anchors(model):
    from label_union.circuit_union import hub_anchors
    return hub_anchors(model)


def basis(anchors, k):
    """Top-k right singular vectors of the (unit) anchor embeddings: k x d."""
    return np.linalg.svd(np.asarray(anchors, float), full_matrices=False)[2][:k]


def project(kw_rows, k, anchors=None, model=None):
    """One client's {label: ('emb', e, r) | other} -> the same rows with k-dimensional e and its r."""
    if anchors is None:
        from label_union import encoder
        anchors = _anchors(model or encoder.DEFAULT_MODEL)
    P = basis(anchors, k)
    Ap = np.asarray(anchors, float) @ P.T
    Ap /= np.linalg.norm(Ap, axis=1, keepdims=True)
    out = {}
    for x, row in kw_rows.items():
        if row[0] != 'emb':
            out[x] = row
            continue
        y = P @ (np.asarray(row[1], float) / (1 << FIX))
        y /= np.linalg.norm(y)
        r = np.sort(Ap @ y)[-HUB:].mean()
        out[x] = ('emb', np.round(y * (1 << FIX)).astype(np.int64), int(round(r * (1 << 2 * FIX))))
    return out
