import unittest
from secfl import oprf
from secfl.keydist import KeyDealer

SEED = bytes.fromhex('a3' * 32)
INFO = bytes.fromhex('74657374206b6579')
BLIND = int.from_bytes(bytes.fromhex('64d37aed22a27f5191de1c1d69fadb899d8862b58eb4220029e036ec4c1f6706'), 'little')
VECTORS = [  # RFC 9497 A.1.1, ristretto255-SHA512, OPRF mode
    ('00', '609a0ae68c15a3cf6903766461307e5c8bb2f95e7e6550e1ffa2dc99e412803c',
     '7ec6578ae5120958eb2db1745758ff379e77cb64fe77b0b2d8cc917ea0869c7e',
     '527759c3d9366f277d8c6020418d96bb393ba2afb20ff90df23fb7708264e2f3ab9135e3bd69955851de4b1f9fe8a0973396719b7912ba9ee8aa7d0b5e24bcf6'),
    ('5a' * 17, 'da27ef466870f5f15296299850aa088629945a17d1f5b7f5ff043f76b3c06418',
     'b4cbf5a4f1eeda5a63ce7b77c7d23f461db3fcab0dd28e4e17cecb5c90d02c25',
     'f4a74c9c592497375e796aa837e907b1a045d34306a749db9f34221f7e750cb4f2a6413a6bf6fa5e19ba6348eb673934a722a7ede2e7621306d18951e7cf2c73'),
]


class OPRFTests(unittest.TestCase):
    def test_rfc9497_vectors(self):
        sk, _ = oprf.derive_key_pair(SEED, INFO)
        self.assertEqual(sk.to_bytes(32, 'little').hex(),
                         '5ebcea5ee37023ccb9fc2d2019f9d7737be85591ae8652ffa9ef0f4d37063b0e')
        for inp, blinded, evaluated, output in VECTORS:
            data = bytes.fromhex(inp)
            r, q = oprf.blind(data, BLIND)
            self.assertEqual(q.hex(), blinded)
            e = oprf.blind_evaluate(sk, q)
            self.assertEqual(e.hex(), evaluated)
            self.assertEqual(oprf.finalize(data, r, e).hex(), output)
            self.assertEqual(oprf.evaluate(sk, data).hex(), output)

    def test_psi_payload_releases_only_held_labels(self):
        dealer = KeyDealer(['cat', 'dog', 'ship', 'truck'])
        server = oprf.PayloadServer(dealer.keys)
        client = oprf.PayloadClient(['dog', 'truck', 'unicorn'], max_queries=8)
        q = client.queries()
        self.assertEqual(len(q), 8)                              # padded
        got = client.recover(server.respond(q), server.table())
        self.assertEqual(got, {'dog': dealer.keys['dog'], 'truck': dealer.keys['truck']})

    def test_blinded_queries_unlinkable(self):
        a = oprf.PayloadClient(['dog'], 1).queries()
        b = oprf.PayloadClient(['dog'], 1).queries()
        self.assertNotEqual(a, b)                                # fresh blinds

    def test_bounds(self):
        with self.assertRaises(ValueError):
            oprf.PayloadClient(['a', 'b'], max_queries=1)
        with self.assertRaises(ValueError):
            oprf.PayloadServer({'a': b'12', 'b': b'1'})


if __name__ == '__main__':
    unittest.main()
