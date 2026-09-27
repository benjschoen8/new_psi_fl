import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import secure_cbn
from secfl.cbn_gan import CBNGenerator, TinyCBNGenerator, ClientCBNGAN, rows, set_rows, trunk_state
from secure_main import build_synthetic
from secure_code_no_cluster import TinyLocalDiscriminator, trim_log
from smoke import TinyClassifier

CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4)
DICTIONARY = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']


def go(**kw):
    torch.manual_seed(7); np.random.seed(7)
    clients, spaces, tests = build_synthetic(CONFIG)
    kw = dict(dict(agg='plain', rounds=2, seed=3, domain_check=False), **kw)
    return secure_cbn.run(clients, spaces, tests, lambda k: TinyCBNGenerator(k, 4, 2), TinyLocalDiscriminator,
                          TinyClassifier, CONFIG, DICTIONARY, **kw), spaces, tests


def init_state(U):
    g0 = secure_cbn.seeded(lambda k: TinyCBNGenerator(k, 4, 2))(U)
    return secure_cbn.flatten(trunk_state(g0))[0], rows(g0)


class CBNModelTests(unittest.TestCase):
    def test_row_size_and_split(self):
        g = CBNGenerator(5)
        self.assertEqual(rows(g).shape, (5, 16 + 2 * (256 + 128 + 64)))                  # 912 per label
        self.assertFalse(any(k.endswith(('gamma.weight', 'beta.weight', 'num_batches_tracked')) or k == 'emb.weight'
                             for k in trunk_state(g)))
        self.assertEqual(g(torch.randn(3, 128), torch.tensor([0, 4, 2])).shape, (3, 3, 32, 32))

    def test_set_rows_roundtrip(self):
        g = TinyCBNGenerator(4)
        t = np.random.default_rng(0).normal(size=rows(g).shape)
        set_rows(g, t)
        np.testing.assert_allclose(rows(g), t, atol=1e-6)
        set_rows(g, t[:1] * 0, which=[2])
        np.testing.assert_allclose(rows(g)[2], 0)

    def test_only_trained_labels_rows_move(self):
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(0)
        g, d = TinyCBNGenerator(3), TinyLocalDiscriminator(3)
        gan = ClientCBNGAN(g, d, dict(gen_noise_dim=4), seed=0)
        gan.load_global({k: v for k, v in trunk_state(g).items()}, rows(g))
        x = torch.rand(8, 3, 2, 2) * 2 - 1
        counts = gan.train(DataLoader(TensorDataset(x, torch.full((8,), 1)), batch_size=4))    # only label 1
        dt, dr = gan.update()
        self.assertEqual(counts, {1: 8})
        self.assertGreater(np.abs(dr[1]).sum(), 0)
        np.testing.assert_array_equal(dr[0], 0)                              # labels without samples: untouched
        np.testing.assert_array_equal(dr[2], 0)
        self.assertGreater(np.abs(dt).sum(), 0)                              # the shared trunk moves


class CBNPipelineTests(unittest.TestCase):
    def test_clients_hold_only_their_rows_and_kem_blocks_others(self):
        from secfl import kem
        from secfl.bb import BulletinBoard
        res, spaces, _ = go(agg='secagg', rounds=1)
        Un = res['union']
        idx = Un['index']
        board = BulletinBoard()
        kem.post_generators(board, Un['pks'], 0, {k: {'row': res['table'][k]} for k in range(Un['U'])})
        cat_client = next(i for i, s in enumerate(spaces.values()) if 'cat' in s)
        mine = {idx[cat_client][x]: Un['sks'][cat_client][x] for x in idx[cat_client]}
        got = kem.fetch_generators(board, mine, Un['pks'], 0)
        self.assertEqual(set(got), set(idx[cat_client].values()))            # own rows only
        other = next(k for k in range(Un['U']) if k not in got)
        forged = {other: next(iter(mine.values()))}
        self.assertEqual(kem.fetch_generators(board, forged, Un['pks'], 0), {})

    def test_plain_gefl_union_has_no_keys(self):
        res, _, _ = go(agg='plain', rounds=1)
        self.assertEqual(res['union']['U'], 3)                               # cat, dog, ship
        self.assertIsNone(res['union']['sks'])
        self.assertTrue(res['evaluator']['union_metrics']['exact'])

    def test_secagg_uncompressed_equals_plain_gefl(self):
        from tests.test_long_run_safety import recount
        a, spaces, tests = go(agg='secagg', quantize=False)
        b, _, _ = go(agg='plain', union_result=a['union'])                   # same (random) indices
        self.assertTrue(a['evaluator']['union_metrics']['exact'])
        np.testing.assert_allclose(a['trunk'], b['trunk'], atol=5e-4)       # 2^-24 fixed point, float32 casts
        np.testing.assert_allclose(a['table'], b['table'], atol=5e-4)
        names = {r['cls']: r['label'] for r in a['evaluator']['experimenter_view']}
        self.assertAlmostEqual(a['history'][-1]['accuracy'],
                               recount(a['model'], spaces, tests, [names[k] for k in range(a['union']['U'])]))

    def test_min_holders_threshold(self):
        res, _, _ = go(agg='plain', rounds=1, min_holders=3)               # every label has 2 holders
        T0, R0 = init_state(res['union']['U'])
        np.testing.assert_array_equal(res['table'], R0)                      # no row reaches t = 3
        self.assertEqual(res['history'][0]['rows_below_threshold'], 3)
        self.assertGreater(np.abs(res['trunk'] - T0).sum(), 0)               # trunk: 3 contributors
        res1, _, _ = go(agg='plain', rounds=1, min_holders=1)
        self.assertEqual(res1['history'][0]['rows_updated'], 3)

    def test_resume_is_identical_and_checks_settings(self):
        for kw in (dict(agg='plain'), dict(agg='secagg')):
            with tempfile.TemporaryDirectory() as d:
                first, _, _ = go(rounds=2, checkpoint_dir=d, **kw)
                full, _, _ = go(rounds=3, union_result=first['union'], **kw)
                resumed, _, _ = go(rounds=3, resume=f'{d}/checkpoint_last.pt', checkpoint_dir=d, **kw)
                with self.assertRaisesRegex(ValueError, 'min_holders'):
                    go(rounds=3, resume=f'{d}/checkpoint_last.pt', min_holders=1, **kw)
                self.assertEqual([p.name for p in Path(d, 'clients').iterdir()], ['round_0003.pt'])
            self.assertEqual(resumed['setup']['resumed_from'], 2)
            np.testing.assert_array_equal(resumed['trunk'], full['trunk'])
            np.testing.assert_array_equal(resumed['table'], full['table'])
            self.assertEqual([h['accuracy'] for h in resumed['history']], [h['accuracy'] for h in full['history']])

    def test_outputs_are_split_by_role(self):
        with tempfile.TemporaryDirectory() as d:
            go(agg='secagg', rounds=1, checkpoint_dir=d)
            agg = torch.load(f'{d}/checkpoint_last.pt', weights_only=False)
            cl = torch.load(f'{d}/{agg["clients_file"]}', weights_only=False)
            setup = Path(d, 'setup.json').read_text()
            ev = json.loads(Path(d, 'evaluator', 'union.json').read_text())
        self.assertFalse({'union', 'clients', 'sks', 'index'} & set(agg))    # Aggregator file: no client secrets
        self.assertIn('sks', cl['union'])
        self.assertNotIn('cat', setup)                                       # no label names outside evaluator/
        self.assertNotIn('cat', json.dumps(agg['history'], default=str))
        self.assertIn('cat', json.dumps(ev))

    def test_trim_log_survives_a_half_written_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'metrics.jsonl')
            p.write_text('{"round": 1}\n{"round": 2}\n{"round": 3, "acc')
            trim_log(p, 2)
            self.assertEqual(p.read_text(), '{"round": 1}\n{"round": 2}\n')


class CBNCompressTests(unittest.TestCase):
    def test_encode_decode_mean_over_holders(self):
        from secfl.secagg import run_secagg
        from secfl import compress
        rng = np.random.default_rng(0)
        U, T, P = 3, 20, 6
        it, ir = np.arange(0, T, 2), np.array([1, 4])
        sc_t, sc_r = np.full(T, .1), np.full((U, P), .2)
        ups = {0: {'T': rng.uniform(-.1, .1, T), 0: rng.uniform(-.2, .2, P)},
               1: {'T': rng.uniform(-.1, .1, T), 0: rng.uniform(-.2, .2, P), 2: rng.uniform(-.2, .2, P)},
               2: {}}                                                        # trained nothing: zeros, still uploads
        for fixed, bits, tol in ((False, compress.BITS, 1 / 127), (True, 64, 1e-6)):
            vec = {c: secure_cbn.encode_update(u, U, it, ir, sc_t, sc_r, rng, fixed) for c, u in ups.items()}
            total, _ = run_secagg(vec, threshold=2, modulus_bits=bits, session=b't')
            mt, mr, n = secure_cbn.decode_update(total, U, it, ir, sc_t, sc_r, fixed)
            self.assertEqual(n.tolist(), [2, 2, 0, 1])
            np.testing.assert_allclose(mt, (ups[0]['T'][it] + ups[1]['T'][it]) / 2, atol=.1 * tol)
            self.assertEqual(set(mr), {0, 2})                                # row 1: nobody, not updated
            np.testing.assert_allclose(mr[0], (ups[0][0][ir] + ups[1][0][ir]) / 2, atol=.2 * tol)
            np.testing.assert_allclose(mr[2], ups[1][2][ir], atol=.2 * tol)

    def test_quantized_pipeline_bytes_and_kept_coordinates(self):
        from secfl import compress
        q, _, _ = go(agg='secagg', rounds=1, keep_frac=.5, min_holders=1)
        plain, _, _ = go(agg='plain', rounds=1, min_holders=1, union_result=q['union'])
        T0, R0 = init_state(q['union']['U'])
        U, T, P = R0.shape[0], T0.size, R0.shape[1]
        kt, kr = round(T * .5), round(P * .5)
        b = q['history'][0]['bytes']
        self.assertEqual(b['upload'], 3 * (kt + U * kr + U + 1) * 2)         # 3 clients, 16 bit payload
        self.assertGreater(b['upload_per_client'], b['upload_payload_per_client'])   # + SecAgg control
        it = compress.keep_index(T, .5, 0, b'cbn-trunk')
        rest = np.setdiff1d(np.arange(T), it)
        np.testing.assert_array_equal(q['trunk'][rest], T0[rest])            # not uploaded: keep global
        self.assertGreater(np.abs(q['trunk'][it] - T0[it]).sum(), 0)
        self.assertLess(np.abs(q['trunk'] - plain['trunk']).max(), .5)       # same direction, coarser

    def test_too_many_clients_for_16_bits(self):
        from types import SimpleNamespace
        many = [SimpleNamespace(id=i, train_loader=None) for i in range(secure_cbn.MAX_CLIENTS_16BIT + 1)]
        with self.assertRaisesRegex(ValueError, '16-bit'):
            secure_cbn.run(many, {}, [], None, None, None, {}, [], agg='secagg')


if __name__ == '__main__':
    unittest.main()
