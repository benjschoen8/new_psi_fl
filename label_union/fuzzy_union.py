"""Fuzzy label union with no label dictionary: public anchors, then the exact OPRF union.

Each client, locally and before anything is sent:
  1. embeds its own keyword for each label (any language) with the public cross-lingual encoder;
  2. snaps it to the nearest word of the public anchor vocabulary (anchor_words.txt: the 20k most
     frequent English words from wordfreq (letters only, no numerals), minus words >= 8x more frequent in fr/es/de/it/pt/nl, so
     'auto' or 'tres' cannot capture a foreign keyword; generic, not a label list; the first
     `anchors` are used);
  3. replaces that word by its synonym class: mutual nearest neighbours of the vocabulary with
     cosine >= `merge` ('boat' / 'ship'; pairs only, so no chains), public, identical at every client.
Snapping score (step 2) is CSLS when hub = k > 0: 2 cos(x, a) - mean cos of a to its k nearest anchors,
so 'hub' words that sit close to everything ('dna', 'asian') stop capturing keywords. A client that
declares its language skips anchors more frequent in that language than in English (public wordfreq
list anchor_false_friends.json): French 'chat' must not snap to English 'chat'.
Keywords are normalized first (normalize: NFKC, '3' -> 'three').
A single Latin letter keeps its own text, case kept: letters are written the same in every language,
and 'A' vs 'a' is a visual distinction an encoder does not make ('3' still snaps, so it can meet 'three').
The class id (| image-domain code) is then an ordinary exact label and oprf_union_with_keys runs
unchanged. A keyword whose nearest anchor has cosine < `floor` keeps its own text (it then matches
identical wording only).

Leakage = the exact protocol: Aggregator U only; clients U; nobody learns who holds what, counts, or
how similar two labels are (grouping is local, against public data). Setup cost = the exact union
(+ local encoding). The price of fuzziness is quality: keywords near a class boundary split (recall),
distinct labels snapping to one class merge (precision). tests/fuzzy_threshold.py picks (anchors, merge, floor) by group MCC.
"""
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

PARAMS_FILE = Path(__file__).with_name('fuzzy_params.json')
WORDS_FILE = Path(__file__).with_name('anchor_words.txt')
FALSE_FRIENDS = Path(__file__).with_name('anchor_false_friends.json')
NUMBER_WORDS = 'zero one two three four five six seven eight nine'.split()
DEFAULTS = dict(anchors=20000, merge=.8, floor=.3, hub=0)                 # overridden by fuzzy_params.json


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


def mutual_pairs(A, merge, block=2048):
    """(i, j), i < j: each is the other's nearest anchor and cos >= merge."""
    nn, best = np.empty(len(A), int), np.empty(len(A), np.float32)
    for s in range(0, len(A), block):
        S = A[s:s + block] @ A.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -2                  # not itself
        nn[s:s + len(S)], best[s:s + len(S)] = S.argmax(1), S.max(1)
    i = np.flatnonzero((nn[nn] == np.arange(len(A))) & (best >= merge))
    return np.stack([i, nn[i]], 1)[i < nn[i]]


def anchor_classes(A, merge):
    """Synonym class of every anchor (public). merge >= 1: every word its own class."""
    return np.arange(len(A)) if merge >= 1 else components(len(A), mutual_pairs(A, merge))


def hub_penalty(A, k, block=2048):
    """Mean cosine of every anchor to its k nearest other anchors (0 if k = 0)."""
    if not k:
        return np.zeros(len(A), np.float32)
    out = np.empty(len(A), np.float32)
    for s in range(0, len(A), block):
        S = A[s:s + block] @ A.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -2
        out[s:s + len(S)] = np.partition(S, -k, 1)[:, -k:].mean(1)
    return out


def false_friends(lang, n):
    """Anchor indices < n that are more frequent in lang than in English (public)."""
    ff = json.loads(FALSE_FRIENDS.read_text()).get((lang or 'en')[:2], [])
    return np.array([i for i in ff if i < n], int)


@lru_cache(maxsize=4)
def load_anchors(model, n, merge, hub=0):
    from label_union import encoder
    A = encoder.embed(anchor_words(n), model)
    return A, anchor_classes(A, merge), hub_penalty(A, hub)


def normalize(t):
    """Public text normalization before embedding: NFKC, trimmed, a bare numeral spelled out
    ('3' -> 'three': the encoder does not put numerals next to number words)."""
    t = unicodedata.normalize('NFKC', str(t)).strip()
    return NUMBER_WORDS[int(t)] if len(t) == 1 and t.isascii() and t.isdigit() else t


def client_keys(labels, texts, E, A, cls, floor, domains=None, pen=None, skip=()):
    """Local: {label: exact key}. E = embeddings of this client's keywords (texts); pen = hub
    penalty per anchor (CSLS), skip = anchors this client ignores (its language's false friends)."""
    S = E @ A.T
    score = 2 * S - (0 if pen is None else pen)
    if len(skip):
        score[:, skip] = -np.inf
    near = score.argmax(1)
    out = {}
    for x, t, j, s in zip(labels, texts, near, S[np.arange(len(labels)), near]):
        t = unicodedata.normalize('NFKC', t).strip()
        symbol = len(t) == 1 and t.isascii() and t.isalpha()
        key = f'anchor:{cls[j]}' if s >= floor and not symbol else 'text:' + t
        out[x] = key + (f'|{domains[x]}' if domains else '')
    return out


def local_keys(names, keywords, p, domains=None):
    """Every client's step 1-3 (simulated together; each only touches its own keywords)."""
    from label_union import encoder
    model = p.get('model', encoder.DEFAULT_MODEL)
    A, cls, pen = load_anchors(model, p['anchors'], p['merge'], p.get('hub', 0))
    langs = p.get('langs') or [None] * len(names)                        # each client's own language
    out = []
    for i, labels in enumerate(names):
        texts = [normalize(keywords[i][x]) for x in labels]
        out.append(client_keys(labels, texts, encoder.embed(texts, model), A, cls, p['floor'],
                               domains[i] if domains else None, pen, false_friends(langs[i], len(A))))
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
