"""MP-SPDZ step C == circuit_union.group() (ideal functionality). Skipped unless MPSPDZ is set."""
import os

import numpy as np
import pytest

from label_union.circuit_union import group, FIX

pytestmark = pytest.mark.skipif(not os.environ.get('MPSPDZ'), reason='set MPSPDZ to an MP-SPDZ directory')


def _img(start, nimg=45, k=6):
    v = np.zeros(nimg, np.int64)
    v[[(start + j) % nimg for j in range(k)]] = 1
    return v


VARIANTS = [(1, 'shamir'), (2, 'shamir'), (2, 'atlas')]


def _check(rows, owners, m=None, variant=(1, 'shamir')):
    from label_union.mpspdz_group import mpspdz_group
    ideal, steps = group(rows, 0.10, 2, owners)
    got, st, K = mpspdz_group(rows, owners, 0.10, 2, m=m, version=variant[0], protocol=variant[1])
    assert got == ideal
    assert all((K[a] == K[b]) == (ideal[a] == ideal[b]) for a in range(len(rows)) for b in range(len(rows)))
    return st, steps


@pytest.mark.parametrize('variant', VARIANTS)
def test_chain_needs_several_steps(variant):
    # one name; image windows shifted by 4 overlap by 2 only with the neighbour: a path of 5 rows
    rows = [(('name', b'three'), _img(4 * j)) for j in range(5)]
    owners = [0, 1, 2, 0, 1]
    order = np.argsort(owners, kind='stable')                    # rows grouped by client
    rows, owners = [rows[i] for i in order], [owners[i] for i in order]
    st, steps = _check(rows, owners, m=3, variant=variant)
    assert steps == 4 and st['propagation_steps'] >= 4


@pytest.mark.parametrize('variant', VARIANTS)
def test_random_exact_with_dummies(variant):
    rng = np.random.default_rng(1)
    for seed in range(3):
        rows, owners = [], []
        for c in range(4):
            for _ in range(rng.integers(1, 5)):
                rows.append((('name', b'w%d' % rng.integers(4)), _img(int(rng.integers(0, 12)))))
                owners.append(c)
        _check(rows, owners, m=5, variant=variant)


@pytest.mark.parametrize('variant', VARIANTS)
def test_fuzzy_with_symbols(variant):
    rng = np.random.default_rng(2)
    d = 16
    base = rng.standard_normal((3, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    def emb(g):
        e = base[g] + 0.02 * rng.standard_normal(d)
        e /= np.linalg.norm(e)
        return ('emb', np.round(e * (1 << FIX)).astype(np.int64), int(round(0.3 * (1 << 2 * FIX))))
    rows = [(emb(0), _img(0)), (('sym', 'a'), _img(10)),
            (emb(0), _img(1)), (('sym', 'A'), _img(10)), (('sym', 'a'), _img(11)),
            (emb(1), _img(0)), (('sym', 'a'), _img(30))]
    owners = [0, 0, 1, 1, 1, 2, 2]
    _check(rows, owners, variant=variant)
