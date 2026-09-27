import secrets
import unittest

from align.psi import psi, psi_with_keys, PSISender, PSIReceiver


class ExactPSITests(unittest.TestCase):
    def test_intersection(self):
        X = ['cat', 'dog', 'zero', 'A', 'a']
        Y = ['dog', 'truck', 'A', 'seven', 'cat']
        self.assertEqual(psi(X, Y), ['cat', 'dog', 'A'])           # case-sensitive, receiver order

    def test_empty_intersection_and_padding(self):
        s, r = PSISender(['x', 'y']), PSIReceiver(['p'], max_queries=16)
        q = r.queries()
        self.assertEqual(len(q), 16)                               # sender sees only the bound
        self.assertEqual(r.intersect(s.respond(q), s.tags()), [])

    def test_keys_only_for_intersection(self):
        keys = {L: secrets.token_bytes(16) for L in ('cat', 'dog', 'ship')}
        got = psi_with_keys(['dog', 'horse', 'ship'], keys, max_queries=8)
        self.assertEqual(got, {'dog': keys['dog'], 'ship': keys['ship']})

    def test_validation(self):
        with self.assertRaises(ValueError):
            PSIReceiver(['a', 'b'], 1)
        with self.assertRaises(ValueError):
            PSISender([])


if __name__ == '__main__':
    unittest.main()
