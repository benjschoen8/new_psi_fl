import unittest

import numpy as np

from label_union import oprf_union
from label_union.domain import signature, anchors, domain_code, client_codes, GENERATORS


def fresh(kind, k=16, seed=99, size=32):
    """New synthetic images (other seed than the anchors), in the pipelines' [-1, 1] range."""
    return GENERATORS[kind](k, np.random.default_rng(seed), size) * 2 - 1


class DomainTests(unittest.TestCase):
    def test_codes_of_unseen_images(self):
        A = anchors(32)
        for kind in GENERATORS:
            code, margin = domain_code(fresh(kind), A)
            self.assertEqual(code, kind)
            self.assertGreater(margin, .05)

    def test_signature_ignores_rotation_flip_transpose(self):
        x = fresh('strokes', 8)
        s = signature(x)
        for t in (np.rot90(x, 1, (2, 3)), x[..., ::-1], np.swapaxes(x, 2, 3)):
            np.testing.assert_allclose(signature(np.ascontiguousarray(t)), s, atol=1e-9)

    def test_grayscale_and_other_sizes(self):
        g = fresh('strokes', 8, size=28)[:, :1]                            # 1 channel, 28x28 (MNIST-like)
        self.assertEqual(domain_code(g)[0], 'strokes')
        self.assertEqual(domain_code(fresh('photo', 8, size=64))[0], 'photo')

    def test_union_splits_same_name_different_kind(self):
        clients = [['cat', '3'], ['cat', '3'], ['cat']]
        domains = [{'cat': 'photo', '3': 'strokes'}, {'cat': 'strokes', '3': 'strokes'}, {'cat': 'photo'}]
        index, U, _ = oprf_union(clients, bucket_bits=16, domains=domains)
        self.assertEqual(U, 3)                                             # cat@photo, cat@strokes, 3@strokes
        self.assertEqual(index[0]['cat'], index[2]['cat'])
        self.assertNotEqual(index[0]['cat'], index[1]['cat'])
        self.assertEqual(index[0]['3'], index[1]['3'])

    def test_client_codes_fallback_for_labels_without_samples(self):
        codes, margin = client_codes({'3': fresh('strokes', 8)}, ['3', '7'])        # '7' has no samples
        self.assertEqual(codes, {'3': 'strokes', '7': 'strokes'})
        self.assertGreater(margin, .05)


if __name__ == '__main__':
    unittest.main()
