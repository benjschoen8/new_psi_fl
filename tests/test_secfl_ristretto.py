import unittest
from secfl import ristretto as r

# RFC 9496 Appendix A.1: encodings of k*B, k = 0..15
MULTIPLES = """
0000000000000000000000000000000000000000000000000000000000000000
e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76
6a493210f7499cd17fecb510ae0cea23a110e8d5b901f8acadd3095c73a3b919
94741f5d5d52755ece4f23f044ee27d5d1ea1e2bd196b462166b16152a9d0259
da80862773358b466ffadfe0b3293ab3d9fd53c5ea6c955358f568322daf6a57
e882b131016b52c1d3337080187cf768423efccbb517bb495ab812c4160ff44e
f64746d3c92b13050ed8d80236a7f0007c3b3f962f5ba793d19a601ebb1df403
44f53520926ec81fbd5a387845beb7df85a96a24ece18738bdcfa6a7822a176d
903293d8f2287ebe10e2374dc1a53e0bc887e592699f02d077d5263cdd55601c
02622ace8f7303a31cafc63f8fc48fdc16e1c8c8d234b2f0d6685282a9076031
20706fd788b2720a1ed2a5dad4952b01f413bcf0e7564de8cdc816689e2db95f
bce83f8ba5dd2fa572864c24ba1810f9522bc6004afe95877ac73241cafdab42
e4549ee16b9aa03099ca208c67adafcafa4c3f3e4e5303de6026e3ca8ff84460
aa52e000df2e16f55fb1032fc33bc42742dad6bd5a8fc0be0167436c5948501f
46376b80f409b29dc2b5f6f0c52591990896e5716f41477cd30085ab7f10301e
e0c418f7c8d9c4cdd7395b93ea124f3ad99021bb681dfc3302a9d99a2e53e64e""".split()

# RFC 9496 Appendix A.3: 64 uniform bytes -> element
FROM_UNIFORM = [
    ('5d1be09e3d0c82fc538112490e35701979d99e06ca3e2b5b54bffe8b4dc772c14d98b696a1bbfb5ca32c436cc61c16563790306c79eaca7705668b47dffe5bb6',
     '3066f82a1a747d45120d1740f14358531a8f04bbffe6a819f86dfe50f44a0a46'),
    ('f116b34b8f17ceb56e8732a60d913dd10cce47a6d53bee9204be8b44f6678b270102a56902e2488c46120e9276cfe54638286b9e4b3cdb470b542d46c2068d38',
     'f26e5b6f7d362d2d2a94c5d0e7602cb4773c95a2e5c31a64f133189fa76ed61b'),
    ('8422e1bbdaab52938b81fd602effb6f89110e1e57208ad12d9ad767e2e25510c27140775f9337088b982d83d7fcf0b2fa1edffe51952cbe7365e95c86eaf325c',
     '006ccd2a9e6867e6a2c5cea83d3302cc9de128dd2a9a57dd8ee7b9d7ffe02826'),
    ('ac22415129b61427bf464e17baee8db65940c233b98afce8d17c57beeb7876c2150d15af1cb1fb824bbd14955f2b57d08d388aab431a391cfc33d5bafb5dbbaf',
     'f8f0c87cf237953c5890aec3998169005dae3eca1fbb04548c635953c817f92a'),
    ('165d697a1ef3d5cf3c38565beefcf88c0f282b8e7dbd28544c483432f1cec7675debea8ebb4e5fe7d6f6e5db15f15587ac4d4d4a1de7191e0c1ca6664abcc413',
     'ae81e7dedf20a497e10c304a765c1767a42d6e06029758d2d7e8ef7cc4c41179'),
    ('a836e6c9a9ca9f1e8d486273ad56a78c70cf18f0ce10abb1c7172ddd605d7fd2979854f47ae1ccf204a33102095b4200e5befc0465accc263175485f0e17ea5c',
     'e2705652ff9f5e44d3e841bf1c251cf7dddb77d140870d1ab2ed64f1a9ce8628'),
]


class RistrettoTests(unittest.TestCase):
    def test_generator_multiples_rfc9496(self):
        for k, hexenc in enumerate(MULTIPLES):
            self.assertEqual((r.BASE * k).encode().hex(), hexenc, k)
            self.assertEqual(r.Point.decode(bytes.fromhex(hexenc)), r.BASE * k)

    def test_from_uniform_rfc9496(self):
        for i, o in FROM_UNIFORM:
            self.assertEqual(r.from_uniform(bytes.fromhex(i)).encode().hex(), o)

    def test_group_laws(self):
        a, b = r.random_scalar(), r.random_scalar()
        self.assertEqual(r.BASE * a + r.BASE * b, r.BASE * (a + b))
        self.assertEqual((r.BASE * a) * b, (r.BASE * b) * a)
        self.assertEqual(r.BASE * r.L, r.IDENTITY)
        self.assertEqual(r.BASE * a - r.BASE * a, r.IDENTITY)

    def test_rejects_noncanonical(self):
        for bad in (b'\x01' + b'\0' * 31,                                  # negative s
                    (r.P).to_bytes(32, 'little'),                          # s >= p
                    b'\xff' * 32, b'\0' * 31):
            with self.assertRaises(ValueError):
                r.Point.decode(bad)

    def test_expand_message_xmd_rfc9380(self):
        # RFC 9380 K.3 expand_message_xmd(SHA-512), DST QUUX-V01-CS02-with-expander-SHA512-256, msg ""
        out = r.expand_message_xmd(b'', b'QUUX-V01-CS02-with-expander-SHA512-256', 0x20)
        self.assertEqual(out.hex(), '6b9a7312411d92f921c6f68ca0b6380730a1a4d982c507211a90964c394179ba')


if __name__ == '__main__':
    unittest.main()
