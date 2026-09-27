"""Public cross-lingual text encoder for fuzzy label matching (label_union.fuzzy_union).

Every client runs the same public model on its own keywords and on the public anchor vocabulary,
locally; nothing is sent. Embeddings are unit vectors, so cosine similarity = dot product.

The model is downloaded once from Hugging Face by sentence-transformers. Offline machines use the
cache: `python -m tests.fuzzy_threshold` (on a machine with internet) writes
data/encoder/<model>.npz with the embedding of every anchor word and every keyword the experiments
use; copy that file along with the code.
"""
from pathlib import Path

import numpy as np

DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CACHE_DIR = Path('data/encoder')


def _cache_path(model, cache_dir):
    return Path(cache_dir) / (model.replace('/', '__') + '.npz')


def load_cache(model=DEFAULT_MODEL, cache_dir=CACHE_DIR):
    p = _cache_path(model, cache_dir)
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return dict(zip(z['texts'].tolist(), z['emb']))


def embed(texts, model=DEFAULT_MODEL, cache_dir=CACHE_DIR):
    """texts -> (len(texts), d) float32 unit vectors; cached texts need no model."""
    cache = load_cache(model, cache_dir)
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                f'{len(missing)} texts are not in the encoder cache {_cache_path(model, cache_dir)} and '
                f'sentence-transformers is not installed. On a machine with internet: pip install '
                f'sentence-transformers && python -m tests.fuzzy_threshold, then copy data/encoder/ here.') from e
        vecs = SentenceTransformer(model).encode(missing, normalize_embeddings=True, show_progress_bar=len(missing) > 1000,
                                                 batch_size=256)
        cache.update(zip(missing, np.asarray(vecs, np.float32)))
        p = _cache_path(model, cache_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cache)
        tmp = p.with_name(p.stem + '.tmp.npz')
        np.savez(tmp, texts=np.array(keys), emb=np.stack([cache[k] for k in keys]).astype(np.float32))
        tmp.replace(p)
    e = np.array([cache[t] for t in texts], np.float32).reshape(len(texts), -1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)
