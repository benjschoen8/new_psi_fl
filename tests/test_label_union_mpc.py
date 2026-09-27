import unittest

from align.mpc import MPC
from label_union.mpc_union import exact_union, exact_union_with_keys
from secfl import ristretto as rg

E3 = MPC(3, b'test-union')


class ExactUnionTests(unittest.TestCase):
    def check(self, clients, slots, dense=True):
        names = sorted({x for c in clients for x in c})
        by_name = {}
        for labels, got in zip(clients, slots):
            self.assertEqual(set(got), set(labels))              # each client learns only its own
            for name, s in got.items():
                self.assertEqual(by_name.setdefault(name, s), s)  # same name -> same slot
        self.assertEqual(len(set(by_name.values())), len(names))  # distinct names -> distinct slots
        if dense:
            self.assertEqual(set(by_name.values()), set(range(len(names))))

    def test_overlapping_label_spaces(self):
        clients = [['0', '1', 'cat', 'A'], ['1', 'a', 'A'], ['cat', 'dog', '0']]
        slots, stats = exact_union(clients, m_max=4, engine=E3)
        self.check(clients, slots)
        self.assertGreater(stats['and_gates'], 0)

    def test_disjoint_and_identical(self):
        self.check([['x'], ['y'], ['z']], exact_union([['x'], ['y'], ['z']], 2, E3)[0])
        same = [['p', 'q']] * 3
        slots, _ = exact_union(same, 2, E3)
        self.assertEqual(slots[0], slots[1])
        self.assertEqual(slots[1], slots[2])

    def test_keys_same_for_holders_and_match_public_key(self):
        clients = [['0', 'cat', 'A'], ['cat', 'dog'], ['A', 'dog', '9']]
        slots, keys, pks, stats = exact_union_with_keys(clients, m_max=3, engine=E3)
        self.check(clients, slots)
        self.assertEqual(len(pks), stats['rows'])                     # one pk per usable slot
        by_name = {}
        for got_slots, got_keys in zip(slots, keys):
            for name, sk in got_keys.items():
                self.assertEqual(by_name.setdefault(name, sk), sk)    # all holders: same sk
                self.assertEqual((rg.BASE * sk).encode(), pks[got_slots[name]])
        self.assertEqual(len(set(by_name.values())), 5)               # distinct labels, distinct keys

    def test_hide_count_permutes_slots(self):
        clients = [['0', '1'], ['1', '2'], ['0', '3']]              # 4 distinct names, 8 slots
        seen = set()
        for _ in range(3):
            slots, stats = exact_union(clients, m_max=2, engine=E3, hide_count=True)
            self.check(clients, slots, dense=False)
            values = {v for c in slots for v in c.values()}
            self.assertTrue(all(0 <= v < stats['rows'] for v in values))
            seen |= values
        self.assertTrue(any(v >= 4 for v in seen))                 # not confined to dense 0..3

    def test_hide_count_with_keys(self):
        clients = [['a', 'b'], ['b', 'c'], ['c', 'a']]
        slots, keys, pks, _ = exact_union_with_keys(clients, m_max=2, engine=E3, hide_count=True)
        self.check(clients, slots, dense=False)
        for got_slots, got_keys in zip(slots, keys):
            for name, sk in got_keys.items():
                self.assertEqual((rg.BASE * sk).encode(), pks[got_slots[name]])

    def test_public_slot_bound(self):
        clients = [['a', 'b', 'c'], ['b', 'd'], ['e']]             # 5 distinct, rows = 16
        slots, stats = exact_union(clients, 3, E3, hide_count=True, max_labels=8)
        self.check(clients, slots, dense=False)
        self.assertEqual(stats['slots'], 8)
        self.assertTrue(all(v < 8 for c in slots for v in c.values()))
        with self.assertRaises(ValueError):
            exact_union(clients, 3, E3, max_labels=4)              # 5 labels do not fit in 4 slots
        slots, keys, pks, _ = exact_union_with_keys(clients, 3, E3, hide_count=True, max_labels=8)
        self.assertEqual(len(pks), 8)
        self.assertEqual((rg.BASE * keys[1]['d']).encode(), pks[slots[1]['d']])

    def test_validation(self):
        with self.assertRaises(ValueError):
            exact_union([['a', 'a'], ['b'], ['c']], 2, E3)
        with self.assertRaises(ValueError):
            exact_union([['a', 'b', 'c'], ['b'], ['c']], 2, E3)
        with self.assertRaises(ValueError):
            exact_union([['a'], ['b']], 2, E3)                    # engine has 3 parties


if __name__ == '__main__':
    unittest.main()
