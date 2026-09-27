"""Phase 3d: server update from the unmasked aggregate.

theta_L <- theta_L - lr * (sum_i n_{i,L} g_{i,L}) / N_L, only if |Group_L| >= t.
Labels below the threshold keep their parameters this round.

GeFL use (clients train generators locally): upload g = -(theta_local - theta_global)
and set lr = 1; this is exactly sample-weighted FedAvg of the per-label generators.
The Aggregator then trains the global classifier from the averaged generators.
"""
import numpy as np


def flatten(state: dict):
    """{name: array} -> (flat float64 vector, spec) with a fixed key order."""
    keys = sorted(state)
    spec = [(k, np.asarray(state[k]).shape, np.asarray(state[k]).dtype) for k in keys]
    flat = np.concatenate([np.asarray(state[k], np.float64).ravel() for k in keys]) if keys else np.zeros(0)
    return flat, spec


def unflatten(flat, spec) -> dict:
    out, off = {}, 0
    for k, shape, dtype in spec:
        n = int(np.prod(shape))
        out[k] = np.asarray(flat[off:off + n]).reshape(shape).astype(dtype)
        off += n
    if off != len(flat):
        raise ValueError('flat vector does not match spec')
    return out


def settle(thetas: dict, grad_sums: dict, N: dict, group_sizes: dict, lr: float, threshold: int):
    """thetas {L: flat params}. Returns (new thetas, sorted list of updated labels)."""
    if lr <= 0 or threshold < 1:
        raise ValueError('lr must be positive and threshold >= 1')
    new, updated = dict(thetas), []
    for L, theta in thetas.items():
        if group_sizes.get(L, 0) >= threshold and N.get(L, 0) > 0:
            g = np.asarray(grad_sums[L], np.float64)
            if g.shape != np.shape(theta):
                raise ValueError(f'{L}: gradient shape mismatch')
            new[L] = np.asarray(theta, np.float64) - lr * g / N[L]
            updated.append(L)
    return new, sorted(updated, key=str)
