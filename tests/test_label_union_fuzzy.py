import tempfile
import unittest
from pathlib import Path

import numpy as np

from label_union.fuzzy_union import anchor_classes, anchor_words, client_keys, components, union


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fake_space(concepts, anchors_of, noise=.05, dim=48, seed=0):
    """Stand-in for the public encoder: every concept has a direction; anchor words and each
    client's keyword (its own language) are that direction plus noise."""
    rng = np.random.default_rng(seed)
    center = {c: rng.standard_normal(dim) for c in concepts}
    words = [w for c in concepts for w in anchors_of[c]]
    A = unit([center[c] + noise * rng.standard_normal(dim) for c in concepts for _ in anchors_of[c]])
    emb = lambda c: unit(center[c] + noise * rng.standard_normal(dim))
    return words, A, emb


def same_partition(a, b):
    pairs = lambda idx: {(i, x, j, y) for i, own in enumerate(idx) for x in own for j, o2 in enumerate(idx)
                         for y in o2 if own[x] == o2[y]}
    return pairs(a) == pairs(b)


class FuzzyAnchorTests(unittest.TestCase):
    def test_anchor_vocabulary_is_generic_and_fixed(self):
        w = anchor_words()
        self.assertEqual(len(w), 20000)
        self.assertEqual(len(set(w)), 20000)
        self.assertTrue({'three', '3', 'cat', 'truck', 'g'} <= set(w))
        self.assertEqual(anchor_words(5), w[:5])

    def test_components(self):
        self.assertEqual(components(5, [(0, 3), (3, 4)]).tolist(), [0, 1, 2, 0, 0])

    def test_synonym_classes(self):
        words, A, _ = fake_space(['three', 'cat'], {'three': ['3', 'three'], 'cat': ['cat']})
        self.assertEqual(anchor_classes(A, .9).tolist(), [0, 0, 1])          # '3' ~ 'three'
        self.assertEqual(anchor_classes(A, 1).tolist(), [0, 1, 2])           # no merging

    def test_keywords_in_any_language_meet_and_floor_falls_back_to_text(self):
        concepts = ['cat', 'dog', 'ship']
        words, A, emb = fake_space(concepts, {c: [c] for c in concepts})
        cls = anchor_classes(A, .9)
        k0 = client_keys(['cat', 'dog'], ['cat', 'dog'], np.stack([emb('cat'), emb('dog')]), A, cls, .3)
        k1 = client_keys(['a', 'b'], ['gato', 'barco'], np.stack([emb('cat'), emb('ship')]), A, cls, .3)
        self.assertEqual(k0['cat'], k1['a'])
        self.assertNotEqual(k0['dog'], k1['b'])
        far = unit(np.linalg.svd(A)[2][-1])                                  # orthogonal to every anchor
        self.assertEqual(client_keys(['x'], ['zorp'], far[None], A, cls, .3), {'x': 'text:zorp'})
        self.assertEqual(client_keys(['x'], ['cat'], emb('cat')[None], A, cls, .3, {'x': 'photo'})['x'],
                         k0['cat'] + '|photo')

    def test_secure_equals_plain_and_kem_access(self):
        from secfl import kem
        from secfl.bb import BulletinBoard
        keys = [{'cat': 'anchor:1', 'Cat': 'anchor:1', 'dog': 'anchor:2'},   # two own labels, one class
                {'gato': 'anchor:1', 'barco': 'anchor:3'},
                {'truck': 'text:truck', 'perro': 'anchor:2'}]
        plain, _, _, Up, _ = union(keys, secure=False)
        index, sks, pks, U, stats = union(keys, secure=True)
        self.assertEqual((U, Up), (4, 4))
        self.assertTrue(same_partition(index, plain))
        self.assertEqual(index[0]['cat'], index[0]['Cat'])
        self.assertGreater(stats['setup_upload_bytes_per_client'], 0)
        rows = np.arange(U * 3.).reshape(U, 3)
        board = BulletinBoard()
        kem.post_generators(board, pks, 0, {g: {'row': rows[g]} for g in range(U)})
        for i in range(3):                                                   # each client opens its own rows
            got = kem.fetch_generators(board, {index[i][x]: sks[i][x] for x in keys[i]}, pks, 0)
            self.assertEqual(set(got), set(index[i].values()))

    def test_encoder_cache_works_offline(self):
        from label_union import encoder
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d, 'm.npz'), texts=np.array(['a cat', 'un gato']), emb=np.array([[3., 4.], [4., 3.]], np.float32))
            np.testing.assert_allclose(encoder.embed(['a cat', 'un gato'], 'm', d), [[.6, .8], [.8, .6]], rtol=1e-6)
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                with self.assertRaisesRegex(RuntimeError, 'fuzzy_threshold'):
                    encoder.embed(['uncached'], 'm', d)


if __name__ == '__main__':
    unittest.main()
