"""Public label codes over a public dictionary (exact labels, no OPRF, no index hashing).

The dictionary is a public, agreed list of canonical label ids (synonyms already folded,
e.g. WordNet synsets). Every client names its classes by these ids locally.
  code(L)   = code_vector(SHA-256("label-code/" || L), dim)   -- anyone can compute it
The label union list itself is built by label_union (label_union.discover).
Rounds then aggregate only the shared trunk; no per-label position is needed there.
"""
import hashlib

import numpy as np

from .code_gan import code_vector


def label_seed(label_id: str) -> bytes:
    return hashlib.sha256(b'label-code/' + str(label_id).encode('utf-8')).digest()


def label_code(label_id: str, dim: int) -> np.ndarray:
    return code_vector(label_seed(label_id), dim)


def index_code(index: int, dim: int) -> np.ndarray:
    """Code of union index k (label_union.dict_union): the Aggregator knows only indices."""
    return code_vector(hashlib.sha256(b'label-index/' + str(int(index)).encode()).digest(), dim)
