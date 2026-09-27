import unittest

import numpy as np

from label_union import oprf_union, index_metrics
from label_union.oprf_union import tags, canonical, oprf_union_with_keys
from tests.test_long_run_safety import no_cluster, cluster, recount


class OprfUnionTests(unittest.TestCase):
    def check(self, clients, index, U):
        by_label = {}
        for labels, got in zip(clients, index):
            self.assertEqual(set(got), set(labels))                    # each client: only its own labels
            for x, k in got.items():
                self.assertEqual(by_label.setdefault(x, k), k)          # same label -> same index
        self.assertEqual(U, len({x for c in clients for x in c}))       # Aggregator: |union| only
        self.assertEqual(sorted(by_label.values()), list(range(U)))     # distinct labels -> 0..U-1
        self.assertTrue(index_metrics(clients, index, U, sorted(by_label))['exact'])

    def test_union_without_dictionary(self):
        clients = [['cat', 'dog', 'A'], ['dog', 'ship', 'a'], ['ship', 'cat', '0', 'A'], ['kitten']]
        index, U, stats = oprf_union(clients, bucket_bits=16)
        self.check(clients, index, U)
        self.assertNotEqual(index[0]['A'], index[1]['a'])                # case matters (EMNIST)

    def test_disjoint_identical_and_parallel(self):
        for clients in ([['x'], ['y'], ['z']], [['p', 'q']] * 3, [[str(i) for i in range(20)], ['3', '30']]):
            self.check(clients, *oprf_union(clients, workers=2, bucket_bits=16)[:2])

    def test_tags_are_keyed_and_consistent(self):
        clients = [['cat', 'dog'], ['dog'], ['cat']]
        t1, t2 = tags(clients), tags(clients)
        self.assertEqual(t1[0]['cat'], t1[2]['cat'])                     # equal labels, equal tags
        self.assertNotEqual(t1[0]['cat'], t1[0]['dog'])
        self.assertNotEqual(t1[0]['cat'], t2[0]['cat'])                  # fresh keys: unlinkable across runs
        self.assertEqual(canonical('ｃａｔ '), canonical('cat'))          # NFKC width + whitespace folded

    def test_collisions_are_detected_and_retried(self):
        # 3 bucket bits, 6 labels: collisions (inside one client and across clients) are near certain
        clients = [['a', 'b', 'c'], ['d', 'e'], ['f', 'a']]
        tries = []
        for _ in range(5):
            index, U, stats = oprf_union(clients, bucket_bits=3, max_tries=200)
            self.check(clients, index, U)                                # never a silent merge
            tries.append(stats['tries'])
        self.assertGreater(max(tries), 1)
        with self.assertRaises(RuntimeError):                            # impossible: 2 buckets, 6 labels
            oprf_union(clients, bucket_bits=1, max_tries=3)

    def test_purity_check_catches_a_cross_client_collision(self):
        from label_union.oprf_union import check
        import numpy as np
        r1, r2 = np.uint64(12345), np.uint64(67891)
        with np.errstate(over='ignore'):
            A, B = r1 + r2, r1 * check(b'x') + r2 * check(b'y')
            self.assertNotEqual(B, check(b'x') * A)
            self.assertEqual(r1 * check(b'x') * np.uint64(3), check(b'x') * (r1 * np.uint64(3)))

    def test_too_many_labels(self):
        with self.assertRaises(ValueError):
            tags([['a', 'b'], ['c']], m_max=1)


class OprfPipelineTests(unittest.TestCase):
    def test_no_cluster(self):
        result, spaces, tests = no_cluster(union='oprf', rounds=2)
        setup = result['setup']
        self.assertEqual((setup['union']['method'], setup['labels']), ('oprf', 3))
        self.assertTrue(setup['union_metrics']['exact'])
        names = {r['cls']: r['label'] for r in setup['experimenter_view']}
        self.assertAlmostEqual(result['history'][-1]['accuracy'],
                               recount(result['model'], spaces, tests, [names[k] for k in range(3)]))

    def test_cluster(self):
        result, spaces, tests = cluster(union='oprf', cluster='private', workers=2)
        setup = result['setup']
        self.assertEqual(setup['groups'], 2)
        self.assertTrue(all(isinstance(k, int) for v in setup['aggregator_view'].values() for k in v))
        self.assertTrue(setup['union_metrics']['exact'])
        names = {r['cls']: r['label'] for r in setup['union_view']}
        self.assertAlmostEqual(result['history'][-1]['accuracy'],
                               recount(result['model'], spaces, tests, [names[k] for k in range(4)]))


class CountHidingTests(unittest.TestCase):
    def test_bucket_word_is_not_forced_odd(self):
        from label_union.oprf_union import _upload
        par = {int(_upload({'x': b'tag'}, 4, 0)[np.flatnonzero(_upload({'x': b'tag'}, 4, 0))[0]] & 1) for _ in range(64)}
        self.assertEqual(par, {0, 1})                                   # parity carries no holder count

    def test_decode_pk_any_number_of_holders(self):
        import secrets
        from label_union.oprf_union import decode_pk, PK_CHUNKS
        pk = [secrets.randbits(16) for _ in range(PK_CHUNKS)]
        M = 1 << 64
        for holders in (1, 2, 3, 300):
            rs = [secrets.randbits(64) for _ in range(holders)]
            A = sum(rs) % M
            row = [A] + [sum(r * c for r in rs) % M for c in pk]
            got = decode_pk(row)
            if got is not None:                                          # None only if 2^49 | A
                self.assertEqual(got, np.array(pk, '>u2').tobytes())
        A = 12 << 20                                                     # even A: still exact
        self.assertEqual(decode_pk([A] + [c * A % M for c in pk]), np.array(pk, '>u2').tobytes())
        self.assertIsNone(decode_pk([A] + [(c * A + 1) % M for c in pk]))   # disagreeing holders

    def test_pks_match_secret_keys(self):
        from secfl import ristretto as rg
        idx, sks, pks, _ = oprf_union_with_keys([['a', 'b'], ['b', 'c'], ['a']], bucket_bits=12)
        for own_i, own_k in zip(idx, sks):
            for x in own_i:
                self.assertEqual(pks[own_i[x]], (rg.BASE * own_k[x]).encode())


if __name__ == '__main__':
    unittest.main()
