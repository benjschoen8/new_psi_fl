import secrets
import unittest

import numpy as np

from align.circuit_psi import circuit_psi
from align.twopc import TwoPC

PC = TwoPC(b'test')     # base OTs once for the whole test module


class CircuitPSITests(unittest.TestCase):
    def test_membership_is_shared_not_revealed(self):
        X = ['cat', 'dog', 'zero', 'A', 'a']
        Y = ['dog', 'truck', 'A', 'seven']
        out = circuit_psi(X, Y, pc=PC)
        m = out['member']
        np.testing.assert_array_equal(PC.open(m), [0, 1, 0, 1, 0])

    def test_key_version(self):
        X = ['dog', 'horse', 'ship']
        Y = ['cat', 'ship', 'dog']
        keys = [secrets.token_bytes(16) for _ in Y]
        out = circuit_psi(X, Y, keys, pc=PC)
        self.assertEqual(out['keys'][0], keys[2])        # dog
        self.assertEqual(out['keys'][2], keys[1])        # ship
        self.assertNotIn(out['keys'][1], keys)           # horse: random, not any real key
        self.assertEqual(len(out['keys'][1]), 16)

    def test_label_sized_sets(self):
        X = [str(i) for i in range(62)]
        Y = [str(i) for i in range(30, 92)]
        out = circuit_psi(X, Y, pc=PC)
        truth = np.array([int(i >= 30) for i in range(62)])
        np.testing.assert_array_equal(PC.open(out['member']), truth)
        for share in (out['member'].s0, out['member'].s1):      # neither share equals the answer
            self.assertGreater(int((share != truth).sum()), 10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            circuit_psi(['a', 'a'], ['b'], pc=PC)
        with self.assertRaises(ValueError):
            circuit_psi(['a'], ['b', 'c'], [b'12', b'1'], pc=PC)


if __name__ == '__main__':
    unittest.main()
