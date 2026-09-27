"""Label union list by one SecAgg of dictionary indicator vectors (the pipelines' method).

Each client uploads a 0/1 vector over the public dictionary (1 = "I hold this label").
SecAgg reveals only the sum: the Aggregator learns which labels exist and how many clients
hold each, never who. Clients learn nothing about each other.
"""
import numpy as np

from secfl.secagg import run_secagg


def indicator(labels, dictionary) -> np.ndarray:
    pos = {x: i for i, x in enumerate(dictionary)}
    missing = [x for x in labels if x not in pos]
    if missing:
        raise ValueError(f'labels not in the public dictionary: {missing}')
    v = np.zeros(len(dictionary), np.uint64)
    v[[pos[x] for x in labels]] = 1
    return v


def discover(client_labels, dictionary, threshold=None):
    """One-time SecAgg -> (existing ids in dictionary order, holder counts, stats)."""
    if len(set(dictionary)) != len(dictionary):
        raise ValueError('dictionary ids must be unique')
    n = len(client_labels)
    threshold = threshold or max(2, -(-2 * n // 3))
    total, _ = run_secagg({i: indicator(l, dictionary) for i, l in enumerate(client_labels)},
                          threshold, modulus_bits=64, session=b'label-discovery')
    existing = [x for x, c in zip(dictionary, total) if c]
    counts = [int(c) for c in total if c]
    return existing, counts, dict(dictionary_size=len(dictionary), upload_bytes_per_client=len(dictionary) * 8)
