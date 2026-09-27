import unittest
from dataclasses import FrozenInstanceError
from secfl.bb import BulletinBoard, PublicParams, Identity, lookup_identity


class BulletinBoardTests(unittest.TestCase):
    def test_append_only_full_read_and_chain(self):
        bb = BulletinBoard()
        h0 = bb.head()
        self.assertEqual(bb.post('agg', 't', b'x'), 0)
        self.assertEqual(bb.post('c1', 't', b'y'), 1)
        entries = bb.read()
        self.assertEqual([e.payload for e in entries], [b'x', b'y'])
        self.assertNotEqual(bb.head(), h0)
        self.assertTrue(bb.verify())
        with self.assertRaises(FrozenInstanceError):
            entries[0].payload = b'z'
        self.assertIsInstance(entries, tuple)   # caller cannot append to the board
        with self.assertRaises(TypeError):
            bb.post('c1', 't', 'not bytes')

    def test_tamper_detected(self):
        bb = BulletinBoard()
        bb.post('a', 't', b'1'); bb.post('a', 't', b'2')
        from dataclasses import replace
        bb._entries[0] = replace(bb._entries[0], payload=b'evil')
        self.assertFalse(bb.verify())

    def test_params_validation(self):
        p = PublicParams(labels=('cat', 'dog'), threshold=2)
        self.assertEqual(p.modulus_bits, 32)
        for bad in (dict(labels=()), dict(labels=('a', 'a')), dict(labels=('a',), clip=0),
                    dict(labels=('a',), frac_bits=31), dict(labels=('a',), threshold=0)):
            with self.assertRaises(ValueError):
                PublicParams(**bad)

    def test_identity_publish_lookup_and_dh(self):
        bb = BulletinBoard()
        a, b = Identity('a'), Identity('b')
        a.publish(bb); b.publish(bb)
        self.assertEqual(lookup_identity(bb, 'a'), a.public_bytes)
        self.assertEqual(a.exchange(lookup_identity(bb, 'b')), b.exchange(lookup_identity(bb, 'a')))
        Identity('a').publish(bb)              # second, different key for 'a'
        with self.assertRaises(LookupError):
            lookup_identity(bb, 'a')
        with self.assertRaises(LookupError):
            lookup_identity(bb, 'nobody')


if __name__ == '__main__':
    unittest.main()
