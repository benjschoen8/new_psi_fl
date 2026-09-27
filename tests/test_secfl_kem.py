import unittest

import numpy as np

from secfl import ristretto as rg
from secfl.bb import BulletinBoard
from secfl.kem import encaps, decaps, seal, unseal, post_generators, fetch_generators


def keypair():
    sk = rg.random_scalar()
    return sk, (rg.BASE * sk).encode()


def state(seed):
    rng = np.random.default_rng(seed)
    return {'w': rng.normal(size=(3, 2)).astype(np.float32), 'b': rng.normal(size=3)}


class KEMTests(unittest.TestCase):
    def test_encaps_decaps(self):
        sk, pk = keypair()
        enc, k = encaps(pk, b'info')
        self.assertEqual(decaps(sk, enc, pk, b'info'), k)
        self.assertNotEqual(decaps(rg.random_scalar(), enc, pk, b'info'), k)
        self.assertNotEqual(encaps(pk, b'info')[1], k)             # fresh k every time

    def test_seal_binding(self):
        sk, pk = keypair()
        blob = seal(pk, 2, 7, state(0))
        np.testing.assert_array_equal(unseal(sk, pk, 2, 7, blob)['w'], state(0)['w'])
        self.assertIsNone(unseal(sk, pk, 3, 7, blob))
        self.assertIsNone(unseal(sk, pk, 2, 8, blob))
        self.assertIsNone(unseal(rg.random_scalar(), pk, 2, 7, blob))
        self.assertIsNone(unseal(sk, pk, 2, 7, blob[:-1] + bytes([blob[-1] ^ 1])))

    def test_board_padded_and_holder_only(self):
        bb, pairs = BulletinBoard(), [keypair() for _ in range(6)]
        pks = [pk for _, pk in pairs]
        post_generators(bb, pks, 1, {1: state(1), 4: state(4)})
        posts = [e for e in bb.read() if e.topic.startswith('gen/1/')]
        self.assertEqual(len(posts), 6)
        self.assertEqual(len({len(e.payload) for e in posts}), 1)
        got = fetch_generators(bb, {4: pairs[4][0], 2: rg.random_scalar()}, pks, 1)
        self.assertEqual(set(got), {4})
        np.testing.assert_array_equal(got[4]['b'], state(4)['b'])

    def test_with_union_keys(self):
        from align.mpc import MPC
        from label_union.mpc_union import exact_union_with_keys
        clients = [['cat', 'dog'], ['dog', 'ship']]
        slots, keys, pks, _ = exact_union_with_keys(clients, 2, MPC(2, b'kem-test'), hide_count=True)
        bb = BulletinBoard()
        post_generators(bb, pks, 0, {slots[0]['dog']: state(9)})
        for c in range(2):
            mine = {slots[c][name]: sk for name, sk in keys[c].items()}
            got = fetch_generators(bb, mine, pks, 0)
            self.assertIn(slots[c]['dog'], got)                       # both dog holders decrypt
        self.assertEqual(len(fetch_generators(bb, {slots[0]['cat']: keys[0]['cat']}, pks, 0)), 1)  # dummy opens as zeros


if __name__ == '__main__':
    unittest.main()
