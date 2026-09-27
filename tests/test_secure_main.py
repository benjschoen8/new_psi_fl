import unittest

import numpy as np

from secure_main import run, build_synthetic, ideal_union_with_keys, aggregate
from secfl.bb import PublicParams
from secfl.upload import UploadLayout
from smoke import TinyGenerator, TinyDiscriminator, TinyClassifier
from secfl import ristretto as rg

CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4,
              global_samples_per_class=4)


class SecureMainTests(unittest.TestCase):
    def go(self, **kw):
        clients, spaces, tests = build_synthetic(CONFIG)
        return run(clients, spaces, tests, lambda: TinyGenerator(1), TinyDiscriminator, TinyClassifier,
                   CONFIG, rounds=2, **kw)

    def test_ideal_union_outputs(self):
        slots, keys, pks, st = ideal_union_with_keys([['a', 'b'], ['b', 'c']], 2)
        self.assertEqual(slots[0]['b'], slots[1]['b'])
        self.assertEqual(len(pks), st['rows'])
        for s, k in zip(slots, keys):
            for x in s:
                self.assertEqual((rg.BASE * k[x]).encode(), pks[s[x]])

    def test_secagg_aggregate_equals_plain(self):
        rng = np.random.default_rng(0)
        updates = {0: ({1: rng.normal(size=5), 3: rng.normal(size=5)}, {1: 4, 3: 2}),
                   1: ({1: rng.normal(size=5)}, {1: 6}),
                   2: ({0: rng.normal(size=5), 3: rng.normal(size=5)}, {0: 3, 3: 5})}
        layout = UploadLayout(PublicParams(labels=tuple(range(4)), modulus_bits=64, frac_bits=24, clip=1e3),
                              {s: 5 for s in range(4)}, n_max=10, num_clients=3)
        gp, Np, Gp, _ = aggregate(updates)
        gs, Ns, Gs, _ = aggregate(updates, layout)
        self.assertEqual({k: v for k, v in Ns.items() if v}, Np)
        self.assertEqual({k: v for k, v in Gs.items() if v}, Gp)
        for s in gp:
            np.testing.assert_allclose(gs[s], gp[s], atol=1e-5)

    def test_plain_and_secagg_pipelines_run(self):
        import torch
        torch.manual_seed(0)
        plain = self.go(agg='plain')
        self.assertEqual(plain['history'][-1]['active_slots'], 3)       # cat, dog, ship
        secure = self.go(agg='secagg')
        self.assertEqual(secure['history'][-1]['active_slots'], 3)
        self.assertIn('broadcast', secure['history'][1]['bytes'])       # KEM downlink ran in round 2
        self.assertTrue(0 <= secure['history'][-1]['accuracy'] <= 1)

    def test_mpc_union_backend(self):
        result = self.go(agg='secagg', union='mpc')
        self.assertEqual(result['history'][-1]['active_slots'], 3)
        self.assertGreater(result['setup']['and_gates'], 0)

    def test_oprf_union_backend(self):
        result = self.go(agg='secagg', union='oprf')
        self.assertEqual(result['setup']['slots'], 3)                   # dense indices: M = U = 3
        self.assertEqual(result['history'][-1]['active_slots'], 3)
        self.assertEqual(result['slots'][0]['dog'], result['slots'][1]['dog'])

    def test_oprf_keys_open_only_own_slots(self):
        from label_union.oprf_union import oprf_union_with_keys
        from secfl import kem
        from secfl.bb import BulletinBoard
        clients = [['A', 'B', 'C'], ['D', 'A'], ['D']]
        slots, keys, pks, _ = oprf_union_with_keys(clients, bucket_bits=16)
        for sl, ks in zip(slots, keys):
            for x in sl:
                self.assertEqual((rg.BASE * ks[x]).encode(), pks[sl[x]])      # pk reached the Aggregator
        bb = BulletinBoard()
        states = {k: {'w': np.full(3, float(k))} for k in range(len(pks))}
        kem.post_generators(bb, pks, 0, states)
        got = kem.fetch_generators(bb, {slots[0][x]: keys[0][x] for x in slots[0]}, pks, 0)
        self.assertEqual(set(got), {slots[0][x] for x in 'ABC'})          # client with A, B, C: not D
        stolen = {slots[1]['D']: keys[0]['A']}                             # own key tried on D's slot
        self.assertEqual(kem.fetch_generators(bb, stolen, pks, 0), {})

    def test_threshold_blocks_single_holder_slots(self):
        result = self.go(agg='plain', threshold=3)                    # every label has 2 holders
        self.assertEqual(result['history'][-1]['active_slots'], 0)


if __name__ == '__main__':
    unittest.main()
