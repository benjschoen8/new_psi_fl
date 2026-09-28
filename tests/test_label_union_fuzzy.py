import tempfile
import unittest
from pathlib import Path

import numpy as np

from label_union.fuzzy_union import (anchor_classes, anchor_words, client_keys, components, false_friends,
                                     hub_penalty, normalize, union)


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
        self.assertTrue({'three', 'zero', 'cat', 'car', 'truck'} <= set(w))
        self.assertFalse({'auto', 'tres', 'de'} & set(w))
        self.assertTrue(all(x.isascii() and x.isalpha() for x in w))                 # words only, no numerals                     # foreign words would capture keywords
        self.assertEqual(anchor_words(5), w[:5])

    def test_components(self):
        self.assertEqual(components(5, [(0, 3), (3, 4)]).tolist(), [0, 1, 2, 0, 0])

    def test_synonym_classes_mutual_nearest_only(self):
        words, A, _ = fake_space(['three', 'cat'], {'three': ['3', 'three'], 'cat': ['cat']})
        self.assertEqual(anchor_classes(A, .9).tolist(), [0, 0, 1])          # '3' ~ 'three'
        self.assertEqual(anchor_classes(A, 1).tolist(), [0, 1, 2])           # no merging
        chain = unit([[1, 0], [.96, .28], [.8, .6], [.6, .8]])               # 0-1 mutual, 2 and 3 not chained in
        self.assertEqual(anchor_classes(chain, .5).tolist(), [0, 0, 1, 1])

    def test_hub_penalty_stops_a_hub_capturing_keywords(self):
        A = unit(np.vstack([np.eye(4), np.ones(4)]))                          # 4 = hub, cos .5 to every word
        x = unit([[1, .4, .4, .4]])                                          # word 0, but a bit closer to the hub
        cls = anchor_classes(A, 1)
        self.assertEqual(client_keys(['x'], ['kw'], x, A, cls, 0)['x'], 'anchor:4')
        self.assertEqual(client_keys(['x'], ['kw'], x, A, cls, 0, pen=hub_penalty(A, 2))['x'], 'anchor:0')

    def test_false_friends_are_skipped_for_their_language(self):
        w = anchor_words()
        chat, cat = w.index('chat'), w.index('cat')
        self.assertIn(chat, false_friends('fr', len(w)))
        self.assertNotIn(chat, false_friends('en', len(w)))
        self.assertNotIn(cat, false_friends('fr', len(w)))
        A = unit([[1, 0], [.9, .44]])                                        # 0 = 'chat', 1 = 'cat'
        k = lambda skip: client_keys(['cat'], ['chat'], A[:1], A, anchor_classes(A, 1), 0, skip=skip)['cat']
        self.assertEqual((k(()), k(np.array([0]))), ('anchor:0', 'anchor:1'))

    def test_normalize_spells_out_numerals(self):
        self.assertEqual([normalize(t) for t in ('3', ' ３ ', 'three', 'Three', 'Cat', '三', 'g', 'G', '10')],
                         ['three', 'three', 'three', 'three', 'cat', '三', 'g', 'G', '10'])

    def test_letters_keep_their_text(self):
        words, A, emb = fake_space(['a'], {'a': ['a']})
        v = np.stack([emb('a'), emb('a')])
        self.assertEqual(client_keys(['A', 'a'], ['A', 'a'], v, A, anchor_classes(A, 1), .3),
                         {'A': 'text:A', 'a': 'text:a'})

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
