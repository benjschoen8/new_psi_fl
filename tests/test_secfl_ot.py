import secrets
import unittest
from secfl import ot


def pairs(n, size=16):
    return [(secrets.token_bytes(size), secrets.token_bytes(size)) for _ in range(n)]


class OTTests(unittest.TestCase):
    def test_base_ot_correct(self):
        p = pairs(8)
        choices = [secrets.randbelow(2) for _ in p]
        self.assertEqual(ot.run_base_ot(p, choices), [m[c] for m, c in zip(p, choices)])

    def test_base_ot_other_message_hidden(self):
        # receiver's key for the unchosen slot does not decrypt it
        p = pairs(4)
        s, r = ot.BaseOTSender(), ot.BaseOTReceiver([0, 1, 0, 1])
        pts = r.msg2(s.msg1())
        cts = s.msg3(pts, p)
        flipped = [(e1, e0) for e0, e1 in cts]
        wrong = r.output(flipped, pts)
        self.assertTrue(all(w != m[1 - c] for w, m, c in zip(wrong, p, [0, 1, 0, 1])))

    def test_iknp_correct_many(self):
        n = 1000
        p = pairs(n, 16)
        choices = [secrets.randbelow(2) for _ in range(n)]
        self.assertEqual(ot.run_iknp(p, choices, b'sess'), [m[c] for m, c in zip(p, choices)])

    def test_iknp_odd_sizes(self):
        p = pairs(13, 5)
        choices = [1] * 13
        self.assertEqual(ot.run_iknp(p, choices), [m[1] for m in p])

    def test_iknp_u_columns_size(self):
        recv, s = ot.IKNPReceiver([0] * 64), ot.IKNPSender()
        pts = s.msg2(recv.msg1()); s.msg4(recv.msg3(pts))
        self.assertEqual(len(recv.msg5()), ot.KAPPA * 8)     # kappa columns of m bits

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            ot.run_base_ot([(b'ab', b'c')], [0])
        with self.assertRaises(ValueError):
            ot.IKNPReceiver([2])
        from secfl import ristretto as rg
        with self.assertRaises(ValueError):
            ot.BaseOTReceiver([0]).msg2(rg.IDENTITY.encode())


if __name__ == '__main__':
    unittest.main()
