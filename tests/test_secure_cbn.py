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
    f = secure_cbn.seeded(lambda k: TinyCBNGenerator(k, 4, 2))
    return secure_cbn.flatten(trunk_state(f(1)))[0], rows(f(U))          # public trunk does not depend on U


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


class PerLabelGeneratorTests(unittest.TestCase):
    def test_one_generator_per_label_nothing_shared(self):
        from torch.utils.data import DataLoader, TensorDataset
        from secfl.cbn_gan import DCGANTemplate, PerLabelGenerator, TinyTemplate
        big = PerLabelGenerator(4, DCGANTemplate(128, 3, (64, 32, 16)))
        self.assertEqual(rows(big).shape, (4, 172800 + 2 * (64 + 32 + 16)))      # conv + BN affine per label
        self.assertEqual(big(torch.randn(3, 128), torch.tensor([0, 3, 0])).shape, (3, 3, 32, 32))
        torch.manual_seed(0)
        g = PerLabelGenerator(3, TinyTemplate(4))
        self.assertTrue((rows(g) == rows(g)[0]).all())                       # same public init for every label
        z, y = torch.randn(5, 4), torch.tensor([2, 0, 2, 1, 0])
        from torch.func import functional_call
        t, sh = g._t[0], g._shapes
        solo = functional_call(t, {n: p.view(s) for (n, s), p in zip(sh, g.emb.weight[2].split([s.numel() for _, s in sh]))}, (z[[0, 2]],))
        torch.testing.assert_close(g(z, y)[[0, 2]], solo)                    # label 2 = its own generator
        gan = ClientCBNGAN(g, TinyLocalDiscriminator(3), dict(gen_noise_dim=4), seed=0)
        gan.load_global(trunk_state(g), rows(g))
        gan.train(DataLoader(TensorDataset(torch.rand(8, 3, 2, 2) * 2 - 1, torch.full((8,), 1)), batch_size=4))
        dt, dr = gan.update()
        np.testing.assert_array_equal(dt, 0)                                 # no shared weights
        np.testing.assert_array_equal(dr[[0, 2]], 0)
        self.assertGreater(np.abs(dr[1]).sum(), 0)

    def test_batched_forward_equals_one_generator_per_label(self):
        from secfl.cbn_gan import DCGANTemplate, PerLabelGenerator
        torch.manual_seed(0)
        g = PerLabelGenerator(9, DCGANTemplate(8, 3, (8, 4, 4)))
        with torch.no_grad():
            g.emb.weight.add_(.05 * torch.randn_like(g.emb.weight))           # rows differ
        z, y = torch.randn(20, 8), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 8] * 2)
        y[0] = 8                                                            # uneven groups
        fast = g._dcgan(z, y)
        out = torch.empty_like(fast)
        for k in y.unique():                                                # <= 4 labels: loop path
            m = y == k
            out[m] = g(z[m], y[m])
        torch.testing.assert_close(fast, out, atol=1e-5, rtol=1e-4)
        ga = torch.autograd.grad(fast.square().sum(), g.emb.weight)[0]
        gb = torch.autograd.grad(sum(g(z[y == k], y[y == k]).square().sum() for k in y.unique()), g.emb.weight)[0]
        torch.testing.assert_close(ga, gb, atol=1e-4, rtol=1e-3)            # float32 rounding

    def test_pipeline_runs(self):
        from secfl.cbn_gan import PerLabelGenerator, TinyTemplate
        torch.manual_seed(7); np.random.seed(7)
        clients, spaces, tests = build_synthetic(CONFIG)
        res = secure_cbn.run(clients, spaces, tests, lambda k: PerLabelGenerator(k, TinyTemplate(4)),
                             TinyLocalDiscriminator, TinyClassifier, CONFIG, DICTIONARY, agg='secagg', rounds=2,
                             seed=3, domain_check=False)
        self.assertEqual(res['table'].shape, (3, 148))
        self.assertTrue(res['evaluator']['union_metrics']['exact'])


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
        b, _, _ = go(agg='plain', quantize=False, union_result=a['union'])   # same (random) indices
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

    def test_warmup_cache_per_client_every_5_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d, 'cache')
            fresh7, _, _ = go(agg='secagg', rounds=1, warmup_epochs=7)                    # no cache
            first, _, _ = go(agg='secagg', rounds=1, warmup_epochs=5, generator_cache=cache,
                             union_result=fresh7['union'])
            self.assertEqual(len(list(cache.glob('generator_*_epochs5_seed3.pt'))), 3)     # one per client
            cont, _, _ = go(agg='secagg', rounds=1, warmup_epochs=7, generator_cache=cache,
                            union_result=fresh7['union'])                                   # loads 5, trains 2
            self.assertEqual(len(list(cache.glob('generator_*_epochs7_seed3.pt'))), 3)
            np.testing.assert_allclose(cont['table'], fresh7['table'], atol=1e-6)          # 5 + 2 == 7
            np.testing.assert_allclose(cont['trunk'], fresh7['trunk'], atol=1e-6)
            stamp = {f: f.stat().st_mtime_ns for f in cache.iterdir()}
            again, _, _ = go(agg='secagg', rounds=1, warmup_epochs=7, generator_cache=cache,
                             union_result=fresh7['union'])                                  # all from cache
            np.testing.assert_allclose(again['table'], fresh7['table'], atol=1e-6)
            go(agg='plain', rounds=1, warmup_epochs=7, generator_cache=cache)               # other union: shared
            self.assertEqual({f: f.stat().st_mtime_ns for f in cache.iterdir()}, stamp)     # nothing retrained
            self.assertFalse(list(cache.glob('*.lock')))

    def test_damaged_or_nan_cache_entries_are_set_aside(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d, 'cache')
            clean, _, _ = go(agg='secagg', rounds=1, warmup_epochs=5, generator_cache=cache)
            files = sorted(cache.glob('generator_*_epochs5_seed3.pt'))
            files[0].write_bytes(b'not a torch file')                     # interrupted copy
            st = torch.load(files[1], weights_only=False)
            k = next(k for k, x in st['gan']['G'].items() if x.is_floating_point())
            st['gan']['G'][k] = st['gan']['G'][k] * float('nan')          # diverged warm-up
            torch.save(st, files[1])
            again, _, _ = go(agg='secagg', rounds=1, warmup_epochs=5, generator_cache=cache,
                             union_result=clean['union'])
            self.assertEqual(len(list(cache.glob('*.bad'))), 2)             # both set aside ...
            self.assertEqual(len(list(cache.glob('generator_*_epochs5_seed3.pt'))), 3)   # ... and re-made
            np.testing.assert_allclose(again['table'], clean['table'], atol=1e-6)    # same result
            np.testing.assert_allclose(again['trunk'], clean['trunk'], atol=1e-6)

    def test_heter_classifier_guides_the_generator(self):
        tiny = lambda cid, k: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, k))
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d, 'cache')
            base, _, _ = go(agg='secagg', rounds=1, warmup_epochs=2, generator_cache=cache)
            zero, _, _ = go(agg='secagg', rounds=1, warmup_epochs=2, generator_cache=cache, guide_factory=tiny,
                            guide_weight=0., union_result=base['union'])
            np.testing.assert_allclose(zero['table'], base['table'], atol=1e-6)   # weight 0 == no guide
            heter, _, _ = go(agg='secagg', rounds=1, warmup_epochs=2, generator_cache=cache, guide_factory=tiny,
                             guide_weight=1., union_result=base['union'])
            self.assertGreater(np.abs(heter['table'] - base['table']).sum(), 0)     # the guide changes G
            g = heter['setup']['guide']
            self.assertEqual((g['from_cache'], len(g['architectures'])), (3, 1))   # reused from `zero`
            self.assertEqual(len(list(cache.glob('classifier_*.pt'))), 3)
            self.assertEqual(len(list(cache.glob('generator_*_epochs2_seed3.pt'))), 9)   # none / w=0 / w=1
            first, _, _ = go(agg='secagg', rounds=1, warmup_epochs=2, generator_cache=cache, guide_factory=tiny,
                             checkpoint_dir=d, union_result=base['union'])
            with self.assertRaisesRegex(ValueError, 'guide'):
                go(agg='secagg', rounds=2, warmup_epochs=2, resume=f'{d}/checkpoint_last.pt')   # not heter

    def test_lock_is_exclusive_and_dead_owner_is_taken_over(self):
        import os
        import socket
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, 'k.lock')
            a, b = secure_cbn._Lock(path), secure_cbn._Lock(path)
            self.assertTrue(a.acquire())
            self.assertFalse(b.acquire())                                           # held by a live process
            a.release()
            path.write_text(f'{socket.gethostname()} 999999999')                   # owner process is gone
            self.assertTrue(b.acquire())
            b.release()
            self.assertFalse(path.exists())

    def test_warmup_moves_round_one_and_resumes(self):
        base, _, _ = go(agg='plain', rounds=1, min_holders=1)
        warm, _, _ = go(agg='plain', rounds=1, min_holders=1, warmup_epochs=3)
        self.assertGreater(np.abs(warm['table'] - base['table']).sum(), 0)          # warm-up changed the update
        self.assertIn('warmup_seconds', warm['setup'])
        with tempfile.TemporaryDirectory() as d:
            full, _, _ = go(agg='secagg', rounds=2, warmup_epochs=2)
            go(agg='secagg', rounds=0, warmup_epochs=2, checkpoint_dir=d, union_result=full['union'])
            ck = torch.load(f'{d}/checkpoint_last.pt', weights_only=False)
            self.assertEqual((ck['round'], ck['warmed']), (0, True))                # saved right after warm-up
            resumed, _, _ = go(agg='secagg', rounds=2, warmup_epochs=2, resume=f'{d}/checkpoint_last.pt',
                               checkpoint_dir=d)
            self.assertNotIn('warmup_seconds', resumed['setup'])                  # not warmed twice
            np.testing.assert_array_equal(resumed['trunk'], full['trunk'])
            np.testing.assert_array_equal(resumed['table'], full['table'])
            with self.assertRaisesRegex(ValueError, 'warmup_epochs'):
                go(agg='secagg', rounds=2, resume=f'{d}/checkpoint_last.pt')        # warm-up setting must match

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


class CBNFuzzyTests(unittest.TestCase):
    """No dictionary: each client names its labels in its own language (fake public encoder)."""
    WORDS = {'cat': ['cat', 'gato', '貓'], 'dog': ['dog', 'perro', '狗'], 'ship': ['ship', 'barco', '船']}

    def setUp(self):
        from unittest import mock
        from label_union import encoder, fuzzy_union
        rng = np.random.default_rng(0)
        center = {c: rng.standard_normal(32) for c in self.WORDS}
        unit = lambda v: v / np.linalg.norm(v)
        vec = {w: unit(center[c] + .05 * rng.standard_normal(32)) for c, ws in self.WORDS.items() for w in ws}
        A = np.array([unit(center[c] + .05 * rng.standard_normal(32)) for c in self.WORDS for _ in (0, 1)])
        self.patches = [mock.patch.object(encoder, 'embed', lambda texts, model=None, cache_dir=None:
                                          np.array([vec[t] for t in texts])),
                        mock.patch.object(fuzzy_union, 'load_anchors', lambda model, n, merge, hub=0:
                                          (A, fuzzy_union.anchor_classes(A, merge), fuzzy_union.hub_penalty(A, hub)))]   # 2 synonyms per concept
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def fuzzy_go(self, **kw):
        _, spaces, _ = build_synthetic(CONFIG)
        keywords = [{x: self.WORDS[x][i % 3] for x in spaces[cid]} for i, cid in enumerate(spaces)]
        return go(union='fuzzy', keywords=keywords, fuzzy=dict(anchors=6, merge=.8, floor=.3), **kw)

    def test_fuzzy_union_across_languages_trains(self):
        res, spaces, _ = self.fuzzy_go(agg='secagg', rounds=2, min_holders=1)
        self.assertEqual(res['union']['U'], 3)                               # cat / gato / 貓 -> one label
        self.assertTrue(res['evaluator']['union_metrics']['exact'])
        self.assertEqual(len(res['history']), 2)
        plain, _, _ = self.fuzzy_go(agg='plain', rounds=1, min_holders=1)   # Plain-GeFL, same grouping
        self.assertTrue(plain['evaluator']['union_metrics']['exact'])

    def test_fuzzy_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            first, _, _ = self.fuzzy_go(agg='secagg', rounds=1, checkpoint_dir=d, min_holders=1)
            full, _, _ = self.fuzzy_go(agg='secagg', rounds=2, union_result=first['union'], min_holders=1)
            resumed, _, _ = self.fuzzy_go(agg='secagg', rounds=2, resume=f'{d}/checkpoint_last.pt',
                                          checkpoint_dir=d, min_holders=1)
        np.testing.assert_array_equal(resumed['table'], full['table'])


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

    def test_clip_feedback_reports_mean_clip_fraction(self):
        from secfl.secagg import run_secagg
        from secfl import compress
        rng = np.random.default_rng(0)
        U, T, P = 2, 8, 4
        it, ir = np.arange(T), np.arange(P)
        sc_t, sc_r = np.full(T, .1), np.full((U, P), .1)
        blocks = (np.array([0] * 4 + [1] * 4), np.array([0, 0, 1, 1]), 2, 2)
        ups = {0: {'T': np.array([.5] * 4 + [0.] * 4), 0: np.array([.5, 0, 0, 0])},     # block 0 all clipped
               1: {'T': np.zeros(T), 1: np.zeros(P)}, 2: {}}
        vec = {c: secure_cbn.encode_update(u, U, it, ir, sc_t, sc_r, rng, False, blocks) for c, u in ups.items()}
        total, _ = run_secagg(vec, threshold=2, modulus_bits=compress.BITS, session=b'c')
        mt, mr, n, (ct, cr) = secure_cbn.decode_update(total, U, it, ir, sc_t, sc_r, False, blocks)
        self.assertEqual(n.tolist(), [2, 1, 1])                               # counts unaffected
        np.testing.assert_allclose(ct, [.5, 0], atol=.01)                     # mean over the 2 trainers
        np.testing.assert_allclose(cr, [.25, 0], atol=.01)

    def test_quantized_pipeline_bytes_and_kept_coordinates(self):
        from secfl import compress
        q, _, _ = go(agg='secagg', rounds=1, keep_frac=.5, min_holders=1)
        plain, _, _ = go(agg='plain', quantize=False, rounds=1, min_holders=1, union_result=q['union'])
        T0, R0 = init_state(q['union']['U'])
        U, T, P = R0.shape[0], T0.size, R0.shape[1]
        kt, kr = round(T * .5), round(P * .5)
        b = q['history'][0]['bytes']
        g0 = secure_cbn.seeded(lambda k: TinyCBNGenerator(k, 4, 2))(U)
        nb = len(secure_cbn.flatten(trunk_state(g0))[1]) + len(secure_cbn.row_spec(g0))   # clip-feedback words
        self.assertEqual(b['upload'], 3 * (kt + U * kr + U + 1 + nb) * 2)    # 3 clients, 16 bit payload
        self.assertGreater(b['upload_per_client'], b['upload_payload_per_client'])   # + SecAgg control
        it = compress.keep_index(T, .5, 0, b'cbn-trunk')
        rest = np.setdiff1d(np.arange(T), it)
        np.testing.assert_array_equal(q['trunk'][rest], T0[rest])            # not uploaded: keep global
        self.assertGreater(np.abs(q['trunk'][it] - T0[it]).sum(), 0)
        corr = np.corrcoef(q['trunk'][it] - T0[it], plain['trunk'][it] - T0[it])[0, 1]
        self.assertGreater(corr, .5)                                         # same direction, coarser

    def test_compressed_plain_is_the_same_numbers_with_less_traffic(self):
        q, _, _ = go(agg='secagg', rounds=2, keep_frac=.5, min_holders=1)
        p, _, _ = go(agg='plain', rounds=2, keep_frac=.5, min_holders=1, union_result=q['union'])
        np.testing.assert_array_equal(p['trunk'], q['trunk'])              # same 8-bit rounding, same mean
        np.testing.assert_array_equal(p['table'], q['table'])
        for a, b in zip(p['history'], q['history']):
            self.assertEqual(a['clip_feedback'], b['clip_feedback'])
            self.assertLess(a['bytes']['upload_per_client'], b['bytes']['upload_payload_per_client'] / 2 + 1)
        T0, R0 = init_state(q['union']['U'])
        g0 = secure_cbn.seeded(lambda k: TinyCBNGenerator(k, 4, 2))(q['union']['U'])
        nb = len(secure_cbn.flatten(trunk_state(g0))[1]) + len(secure_cbn.row_spec(g0))
        kt, kr = round(T0.size * .5), round(R0.shape[1] * .5)
        rows = sum(len(v) for v in go(agg='plain', rounds=0, union_result=q['union'])[1].values())
        self.assertEqual(p['history'][0]['bytes']['upload'], 3 * (kt + nb) + rows * (kr + 2))   # 1 B / value

    def test_downlink_is_8bit_change_and_clients_rebuild_it(self):
        q, _, _ = go(agg='secagg', rounds=3, min_holders=1)
        f, _, _ = go(agg='secagg', rounds=3, min_holders=1, quantize=False, union_result=q['union'])
        p, _, _ = go(agg='plain', rounds=3, min_holders=1, union_result=q['union'])
        np.testing.assert_array_equal(p['table'], q['table'])              # KEM clients rebuilt exactly `sent`
        step = np.abs(q['table'] - q['sent']).max(1)                        # sent trails the table by one
        self.assertTrue((step > 0).any())                                   # round's 8-bit rounding at most
        for a, b in zip(q['history'][1:], f['history'][1:]):                # KEM rows: 1 B + 4 B scale vs 8 B
            self.assertLess(a['bytes']['broadcast'], b['bytes']['broadcast'])  # per value (same trunk blob)

    def test_too_many_clients_for_16_bits(self):
        from types import SimpleNamespace
        many = [SimpleNamespace(id=i, train_loader=None) for i in range(secure_cbn.MAX_CLIENTS_16BIT + 1)]
        with self.assertRaisesRegex(ValueError, '16-bit'):
            secure_cbn.run(many, {}, [], None, None, None, {}, [], agg='secagg')


if __name__ == '__main__':
    unittest.main()


class ClientProcessTests(unittest.TestCase):
    def test_worker_processes_give_the_same_run_as_threads(self):
        from tensor_loader import TensorLoader

        def run(procs, ckpt):
            torch.manual_seed(7); np.random.seed(7)
            clients, spaces, tests = build_synthetic(CONFIG)
            for c in clients:                              # pre-decoded 8-bit images (what the workers need)
                x, y = c.train_loader.dataset.tensors
                u8 = torch.round((x.clamp(-1, 1) * .5 + .5) * 255).to(torch.uint8)
                c.train_loader = TensorLoader.from_tensors(u8, y, 4, shuffle=True)
            return secure_cbn.run(clients, spaces, tests, lambda k: TinyCBNGenerator(k, 4, 2), TinyLocalDiscriminator,
                                  TinyClassifier, CONFIG, DICTIONARY, agg='secagg', rounds=3, seed=3,
                                  domain_check=False, workers=2, client_procs=procs, checkpoint_dir=ckpt,
                                  union_result=union)
        global union
        union = None
        with tempfile.TemporaryDirectory() as d:
            a = run(False, Path(d) / 'threads')
            union = a['union']
            b = run(True, Path(d) / 'procs')
            np.testing.assert_array_equal(a['table'], b['table'])
            np.testing.assert_array_equal(a['trunk'], b['trunk'])
            self.assertEqual([h['accuracy'] for h in a['history']], [h['accuracy'] for h in b['history']])
            ca = torch.load(Path(d) / 'threads' / 'clients' / 'round_0003.pt', weights_only=False)['clients']
            cb = torch.load(Path(d) / 'procs' / 'clients' / 'round_0003.pt', weights_only=False)['clients']
            for cid in ca:                                 # checkpointed client state and shuffle state match
                for k in ('G', 'D'):
                    for n in ca[cid][k]:
                        torch.testing.assert_close(ca[cid][k][n], cb[cid][k][n], rtol=0, atol=0)
                self.assertTrue(torch.equal(ca[cid]['shuffle'], cb[cid]['shuffle']))
