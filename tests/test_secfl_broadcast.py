import secrets
import unittest

import numpy as np

from secfl.bb import BulletinBoard
from secfl.broadcast import seal, unseal, post_generators, fetch_generators, pack_state, unpack_state


def state(seed):
    rng = np.random.default_rng(seed)
    return {'fc.weight': rng.normal(size=(4, 3)).astype(np.float32), 'fc.bias': rng.normal(size=4)}


class BroadcastTests(unittest.TestCase):
    def test_roundtrip_and_binding(self):
        k = secrets.token_bytes(16)
        blob = seal(k, 3, 'cat', state(0))
        out = unseal(k, 3, 'cat', blob)
        self.assertTrue(np.array_equal(out['fc.weight'], state(0)['fc.weight']))
        self.assertEqual(out['fc.weight'].dtype, np.float32)
        self.assertIsNone(unseal(k, 4, 'cat', blob))              # other round
        self.assertIsNone(unseal(k, 3, 'dog', blob))              # other label
        self.assertIsNone(unseal(secrets.token_bytes(16), 3, 'cat', blob))
        tampered = blob[:-1] + bytes([blob[-1] ^ 1])
        self.assertIsNone(unseal(k, 3, 'cat', tampered))

    def test_board_fetch_only_held_labels(self):
        bb = BulletinBoard()
        keys = {L: secrets.token_bytes(16) for L in ('cat', 'dog', 'ship')}
        post_generators(bb, keys, 1, {L: state(i) for i, L in enumerate(keys)})
        mine = {'dog': keys['dog'], 'ship': secrets.token_bytes(16)}   # 'ship' key is a decoy from OT
        got = fetch_generators(bb, mine, 1)
        self.assertEqual(set(got), {'dog'})
        self.assertTrue(np.array_equal(got['dog']['fc.bias'], state(1)['fc.bias']))
        self.assertEqual(fetch_generators(bb, mine, 2), {})

    def test_padding_hides_active_count(self):
        bb = BulletinBoard()
        keys = {slot: secrets.token_bytes(16) for slot in range(8)}
        post_generators(bb, keys, 1, {2: state(0), 5: state(1)})       # 2 active of 8 slots
        posts = [e for e in bb.read() if e.topic.startswith('gen/1/')]
        self.assertEqual(len(posts), 8)
        self.assertEqual(len({len(e.payload) for e in posts}), 1)       # all ciphertexts same length
        got = fetch_generators(bb, {5: keys[5]}, 1)
        self.assertTrue(np.array_equal(got[5]['fc.bias'], state(1)['fc.bias']))
        with self.assertRaises(ValueError):
            post_generators(bb, {0: keys[0]}, 2, {9: state(0)})

    def test_no_pickle(self):
        with self.assertRaises(ValueError):
            unpack_state(pack_state({'x': np.array([object()], dtype=object)}))


if __name__ == '__main__':
    unittest.main()
