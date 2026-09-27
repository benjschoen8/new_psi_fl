import unittest

import numpy as np

from secfl.secagg import run_secagg, SecAggClient

MOD = 2 ** 32


def inputs(n, length=50, seed=0):
    rng = np.random.default_rng(seed)
    return {f'c{i}': rng.integers(0, MOD, length, dtype=np.uint64) for i in range(n)}


def plain_sum(xs, ids):
    return sum((xs[u] for u in ids), np.zeros(len(next(iter(xs.values()))), np.uint64)) % np.uint64(MOD)


class SecAggTests(unittest.TestCase):
    def test_no_dropout(self):
        xs = inputs(5)
        total, U3 = run_secagg(xs, threshold=3)
        self.assertTrue((total == plain_sum(xs, xs)).all())

    def test_dropout_before_masking(self):
        xs = inputs(6, seed=1)
        total, U3 = run_secagg(xs, threshold=3, drop_before_masking={'c1', 'c4'})
        self.assertEqual(set(U3), {'c0', 'c2', 'c3', 'c5'})
        self.assertTrue((total == plain_sum(xs, U3)).all())

    def test_dropout_during_unmask(self):
        xs = inputs(6, seed=2)
        total, U3 = run_secagg(xs, threshold=3, drop_before_masking={'c0'}, drop_before_unmask={'c2', 'c3'})
        self.assertTrue((total == plain_sum(xs, U3)).all())       # c2, c3 inputs still count

    def test_parallel_masking(self):
        xs = inputs(6, seed=5)
        total, U3 = run_secagg(xs, threshold=3, workers=4, drop_before_masking={'c2'})
        self.assertTrue((total == plain_sum(xs, U3)).all())

    def test_too_many_dropouts_abort(self):
        xs = inputs(4, seed=3)
        with self.assertRaises(ValueError):
            run_secagg(xs, threshold=3, drop_before_masking={'c0', 'c1'})

    def test_masked_input_looks_random(self):
        xs = {u: np.zeros(200, np.uint64) for u in ('a', 'b', 'c')}
        clients = {u: SecAggClient(u, list(xs), 2, 200) for u in xs}
        adverts = {u: c.advertise() for u, c in clients.items()}
        shares = {u: c.share(adverts) for u, c in clients.items()}
        inbox = {v: {u: shares[u][v] for u in shares} for v in xs}
        y = clients['a'].masked_input(xs['a'], inbox['a'])
        self.assertGreater(len(np.unique(y)), 190)                 # zero input, masked output spread

    def test_client_refuses_bad_survivor_set(self):
        xs = inputs(3, seed=4)
        clients = {u: SecAggClient(u, list(xs), 2, 50) for u in xs}
        adverts = {u: c.advertise() for u, c in clients.items()}
        shares = {u: c.share(adverts) for u, c in clients.items()}
        inbox = {v: {u: shares[u][v] for u in shares} for v in xs}
        clients['c0'].masked_input(xs['c0'], inbox['c0'])
        with self.assertRaises(ValueError):
            clients['c0'].unmask({'c0', 'ghost'})


if __name__ == '__main__':
    unittest.main()
