"""Fuzzy label union with no label dictionary: public anchors, then the exact OPRF union.

Each client, locally and before anything is sent:
  1. embeds its own keyword for each label (any language) with the public cross-lingual encoder;
  2. snaps it to the nearest word of the public anchor vocabulary (anchor_words.txt: the 20k most
     frequent English words from wordfreq; generic, not a label list; the first `anchors` are used);
  3. replaces that word by its synonym class: connected components of the vocabulary at cosine
     >= `merge` ('3' / 'three'), computed from public data only, identical at every client.
The class id (| image-domain code) is then an ordinary exact label and oprf_union_with_keys runs
unchanged. A keyword whose nearest anchor has cosine < `floor` keeps its own text (it then matches
identical wording only).

Leakage = the exact protocol: Aggregator U only; clients U; nobody learns who holds what, counts, or
how similar two labels are (grouping is local, against public data). Setup cost = the exact union
(+ local encoding). The price of fuzziness is quality: keywords near a class boundary split (recall),
distinct labels snapping to one class merge (precision; the vocabulary is lowercase, so EMNIST 'A'
and 'a' merge). tests/fuzzy_threshold.py picks (anchors, merge, floor) by group MCC.
"""
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

PARAMS_FILE = Path(__file__).with_name('fuzzy_params.json')
WORDS_FILE = Path(__file__).with_name('anchor_words.txt')
DEFAULTS = dict(anchors=20000, merge=.8, floor=.3)                 # overridden by fuzzy_params.json


def params():
    """Public fuzzy-matching parameters: vocabulary size, synonym-merge cosine, snap floor."""
    return dict(DEFAULTS, **(json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}))


def anchor_words(n=None):
    return WORDS_FILE.read_text().split()[:n]


def components(n, pairs):
    """Union-find over n nodes -> component id per node, numbered by smallest member."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(a) for a in range(n)]
    order = {r: g for g, r in enumerate(sorted(set(roots)))}
    return np.array([order[r] for r in roots])


def similar_pairs(A, merge, block=2048):
    """(i, j), i < j, with cos(A_i, A_j) >= merge."""
    out = []
    for s in range(0, len(A), block):
        i, j = np.nonzero(A[s:s + block] @ A.T >= merge)
        keep = i + s < j
        out.append(np.stack([i[keep] + s, j[keep]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def anchor_classes(A, merge):
    """Synonym class of every anchor (public). merge >= 1: every word its own class."""
    return np.arange(len(A)) if merge >= 1 else components(len(A), similar_pairs(A, merge))


@lru_cache(maxsize=4)
def load_anchors(model, n, merge):
    from label_union import encoder
    A = encoder.embed(anchor_words(n), model)
    return A, anchor_classes(A, merge)


def client_keys(labels, texts, E, A, cls, floor, domains=None):
    """Local: {label: exact key}. E = embeddings of this client's keywords (texts)."""
    S = E @ A.T
    near = S.argmax(1)
    out = {}
    for x, t, j, s in zip(labels, texts, near, S[np.arange(len(labels)), near]):
        key = f'anchor:{cls[j]}' if s >= floor else 'text:' + unicodedata.normalize('NFKC', t).strip()
        out[x] = key + (f'|{domains[x]}' if domains else '')
    return out


def local_keys(names, keywords, p, domains=None):
    """Every client's step 1-3 (simulated together; each only touches its own keywords)."""
    from label_union import encoder
    model = p.get('model', encoder.DEFAULT_MODEL)
    A, cls = load_anchors(model, p['anchors'], p['merge'])
    out = []
    for i, labels in enumerate(names):
        texts = [keywords[i][x] for x in labels]
        out.append(client_keys(labels, texts, encoder.embed(texts, model), A, cls, p['floor'],
                               domains[i] if domains else None))
    return out


def union(keys, workers=1, secure=True, session=b'fuzzy-union'):
    """keys: per client {label: key}. Exact union over keys (two labels may share a key: same row).
    Returns index, sks (None if not secure), pks (None), U, stats."""
    uniq = [sorted(set(k.values())) for k in keys]
    if secure:
        from label_union.oprf_union import oprf_union_with_keys
        idx, sk, pks, stats = oprf_union_with_keys(uniq, workers=workers, session=session)
        U = len(pks)
    else:
        order = {v: g for g, v in enumerate(sorted(set().union(*map(set, uniq))))}
        idx, sk, pks, U = [{v: order[v] for v in u} for u in uniq], None, None, len(order)
        stats = dict(setup_upload_bytes_per_client=0, setup_download_bytes_per_client=0)
    index = [{x: idx[i][v] for x, v in k.items()} for i, k in enumerate(keys)]
    sks = None if sk is None else [{x: sk[i][v] for x, v in k.items()} for i, k in enumerate(keys)]
    return index, sks, pks, U, stats
