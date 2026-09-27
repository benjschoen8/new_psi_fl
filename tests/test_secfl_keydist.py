import unittest
from secfl.keydist import KeyDealer, share_bits, release_keys, release_keys_in_circuit, assign_slots


class KeyDistTests(unittest.TestCase):
    def check(self, n_labels):
        labels = [f'L{i}' for i in range(n_labels)]
        dealer = KeyDealer(labels)
        match = [i % 3 == 0 for i in range(n_labels)]
        b_c, b_s = share_bits([int(m) for m in match])
        got = release_keys(dealer, b_s, b_c, b'client-7')
        for L, m in zip(labels, match):
            if m:
                self.assertEqual(got[L], dealer.keys[L])
            else:
                self.assertNotEqual(got[L], dealer.keys[L])
                self.assertEqual(len(got[L]), 16)

    def test_base_ot_path(self):
        self.check(10)

    def test_iknp_path(self):
        self.check(200)

    def test_shares_hide_bits(self):
        # each share alone is uniform: over many runs both values appear for a fixed bit
        seen_c = {share_bits([1])[0][0] for _ in range(64)}
        self.assertEqual(seen_c, {0, 1})

    def test_malicious_client_flips_ot_choices_and_steals_keys(self):
        # Documents the weakness of the separate-OT release: flipping b_c yields K_L
        # exactly for the labels the client does NOT hold.
        labels = ['cat', 'dog', 'ship']
        dealer = KeyDealer(labels)
        b_c, b_s = share_bits([1, 0, 0])                 # client holds only 'cat'
        stolen = release_keys(dealer, b_s, [1 - c for c in b_c])
        self.assertEqual(stolen['dog'], dealer.keys['dog'])
        self.assertEqual(stolen['ship'], dealer.keys['ship'])

    def test_in_circuit_release_only_held(self):
        dealer = KeyDealer(['cat', 'dog', 'ship'])
        got = release_keys_in_circuit(dealer, [1, 0, 0])
        self.assertEqual(got['cat'], dealer.keys['cat'])
        self.assertNotEqual(got['dog'], dealer.keys['dog'])
        self.assertNotEqual(got['ship'], dealer.keys['ship'])
        with self.assertRaises(ValueError):
            release_keys_in_circuit(dealer, [1, 0])

    def test_assign_slots(self):
        slots = assign_slots(range(10), 256)
        self.assertEqual(len(set(slots.values())), 10)
        self.assertTrue(all(0 <= s < 256 for s in slots.values()))
        self.assertNotEqual(sorted(slots.values()), list(range(10)))   # not dense ids (p ~ 1)
        with self.assertRaises(ValueError):
            assign_slots(range(5), 4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            KeyDealer(['a', 'a'])
        d = KeyDealer(['a', 'b'])
        with self.assertRaises(ValueError):
            d.ot_pairs([1])
        with self.assertRaises(ValueError):
            release_keys(d, [0, 1], [0])


if __name__ == '__main__':
    unittest.main()
