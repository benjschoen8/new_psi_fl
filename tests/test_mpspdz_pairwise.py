"""Pairwise setup orchestration and opt-in real MP-SPDZ comparisons."""
import importlib
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def runner():
    assert importlib.util.find_spec('label_union.mpspdz_pairwise'), 'pairwise runner is not implemented'
    return importlib.import_module('label_union.mpspdz_pairwise')


@pytest.mark.parametrize('n', [3, 5, 10, 30, 50])
@pytest.mark.parametrize('cap', [1, 2, 4])
def test_round_robin_covers_every_pair_and_caps_client_degree(n, cap):
    batches = runner()._pair_batches(n, cap)
    pairs = [pair for batch in batches for pair in batch]
    assert sorted(pairs) == list(itertools.combinations(range(n), 2))
    for batch in batches:
        assert all(sum(p in pair for pair in batch) <= cap for p in range(n))


@pytest.mark.parametrize('content', ['S 0 12\nS 0 13\n', 'S 0 12\n', 'S 0 12\nS 2 14\n', 'S 0 12\nS 1 nope\n'])
def test_private_parser_rejects_duplicate_missing_or_malformed_indices(tmp_path, content):
    path = tmp_path / 'private'
    path.write_text(content)
    with pytest.raises(RuntimeError, match='private output') as err:
        runner()._parse_private(path, 'S', 2)
    assert content.strip() not in str(err.value)


def test_private_parser_preserves_signed_field_values_and_index_order(tmp_path):
    path = tmp_path / 'private'
    path.write_text('metadata\nS 1 -37\nS 0 42\n')
    assert runner()._parse_private(path, 'S', 2) == [42, -37]


def test_bridge_keeps_each_endpoint_share_in_global_pair_order(tmp_path):
    # The pairs arrive in deliberately nonlexicographic completion order.
    pairs = {}
    for p, q in [(1, 2), (0, 2), (0, 1)]:
        paths = []
        for endpoint in (p, q):
            path = tmp_path / f'pair{p}{q}-P{endpoint}'
            values = [1000 * p + 100 * q + 10 * endpoint + i for i in range(4)]
            path.write_text(''.join(f'S {i} {value}\n' for i, value in enumerate(values)))
            paths.append(path)
        pairs[p, q] = paths
    prefix = tmp_path / 'graph-Input'
    runner()._bridge_inputs(3, 2, pairs, prefix)
    assert (tmp_path / 'graph-Input-P0-0').read_text().split() == [str(v) for v in [100, 101, 102, 103, 200, 201, 202, 203]]
    assert (tmp_path / 'graph-Input-P1-0').read_text().split() == [str(v) for v in [110, 111, 112, 113, 1210, 1211, 1212, 1213]]
    assert (tmp_path / 'graph-Input-P2-0').read_text().split() == [str(v) for v in [220, 221, 222, 223, 1220, 1221, 1222, 1223]]
    assert all((tmp_path / f'graph-Input-P{p}-0').stat().st_mode & 0o777 == 0o600 for p in range(3))


def test_port_reservations_do_not_overlap_between_threads():
    module = runner()
    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(lambda _: module._reserve_ports(3), range(8)))
    try:
        ports = [port for base in reservations for port in range(base, base + 3)]
        assert len(ports) == len(set(ports))
    finally:
        for base in reservations:
            module._release_ports(base, 3)


@pytest.mark.parametrize('fail_early', [False, True, 'signal'])
def test_session_timeout_or_party_failure_reaps_every_child(tmp_path, fail_early, monkeypatch):
    module = runner()
    script = tmp_path / 'fake-party'
    script.write_text('#!' + sys.executable + '\n'
                      'import pathlib, sys, time\n'
                      'p = int(sys.argv[sys.argv.index("-p") + 1])\n'
                      'pathlib.Path("pid-" + str(p)).write_text(str(__import__("os").getpid()))\n'
                      + ('if p == 1: __import__("os").kill(__import__("os").getpid(), 9)\n'
                         if fail_early == 'signal' else 'if p == 1: sys.exit(7)\n' if fail_early else '') +
                      'time.sleep(60)\n')
    script.chmod(0o700)
    children = []
    popen = subprocess.Popen
    def record_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', record_child)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='party failed|timed out') as err:
        module._run_session_once(tmp_path, str(script), 3, 'unused', tmp_path / 'in', tmp_path / 'out', tmp_path / 'log', timeout=2 if fail_early else .4)
    if fail_early:
        assert 'party=1' in str(err.value)
        assert ('SIGKILL' if fail_early == 'signal' else 'exit=7') in str(err.value)
    assert time.monotonic() - start < 4
    assert len(children) == 3
    for child in children:
        with pytest.raises(ProcessLookupError):
            os.kill(child.pid, 0)


def test_one_pair_failure_cancels_running_siblings(tmp_path, monkeypatch):
    module = runner()
    for name in ('compile.py', 'semi-party.x', 'shamir-party.x'):
        (tmp_path / name).touch()
    monkeypatch.setattr(module, '_compile', lambda *args: ('program', 0, False))
    sibling_started = threading.Event()
    sibling_cancelled = threading.Event()
    calls = []
    def session(*args, cancel=None, **_):
        calls.append(args[6])
        if len(calls) == 1:
            sibling_started.set()
            if cancel is not None and cancel.wait(2):
                sibling_cancelled.set()
            return {}
        assert sibling_started.wait(1)
        raise RuntimeError('deliberate pair failure')
    monkeypatch.setattr(module, '_run_session', session)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='deliberate pair failure'):
        module.mpspdz_pairwise_group([(('name', b'x'), None)] * 3, [0, 1, 2],
                                    root=tmp_path, pair_workers=2)
    assert sibling_cancelled.is_set()
    assert time.monotonic() - start < 1.5
    assert not list((tmp_path / 'Player-Data').iterdir())


@pytest.mark.parametrize('rows,owners,options', [
    ([], [], {}),
    ([(('name', b'x'), None)], [0, 1, 2], {}),
    ([(('name', b'x'), None)] * 3, [0, 1, -1], {}),
    ([(('name', b'x'), None)] * 3, [0, 0, 1], {}),
    ([(('name', b'x'), None)] * 4, [0, 0, 1, 2], {'m': 1}),
    ([(('name', b'x'), None)] * 3, [0, 1, 2], {'pair_concurrency': 0}),
    ([(('name', b'x'), None)] * 3, [0, 1, 2], {'fix': 8}),
])
def test_invalid_inputs_fail_before_backend_lookup(rows, owners, options):
    with pytest.raises(ValueError):
        runner().mpspdz_pairwise_group(rows, owners, root='/nonexistent', **options)


@pytest.mark.skipif(not os.environ.get('MPSPDZ'), reason='set MPSPDZ for actual semi/Shamir integration')
@pytest.mark.parametrize('case', ['exact', 'fuzzy', 'transitive'])
@pytest.mark.parametrize('prefix', ['serial', 'parallel'])
@pytest.mark.parametrize('block_rows', [2, 64])
@pytest.mark.parametrize('grouping', [(1, 'shamir'), (2, 'shamir'), (2, 'atlas')])
def test_real_mpc_matches_ideal_with_padding(case, prefix, block_rows, grouping):
    from label_union.circuit_union import group
    if case == 'fuzzy':
        a = ('emb', np.array([128, 0]), 0)
        b = ('emb', np.array([0, 128]), 0)
        rows = [(a, None), (('sym', 'A'), None), (a, None), (('sym', 'A'), None), (b, None)]
        owners = [0, 0, 1, 2, 2]
    elif case == 'transitive':
        images = [np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])]
        rows = [(('name', b'x'), image) for image in images]
        owners = [0, 1, 2]
    else:
        rows = [(('name', name), None) for name in [b'x', b'y', b'x', b'z', b'z']]
        owners = [0, 0, 1, 1, 2]
    expected, _ = group(rows, .10, 1, owners)
    got, stats, keys = runner().mpspdz_pairwise_group(rows, owners, t=1, m=3, timeout=120,
        pair_workers=2, prefix=prefix, block_rows=block_rows,
        group_version=grouping[0], group_protocol=grouping[1])
    assert got == expected
    assert all((keys[a] == keys[b]) == (got[a] == got[b]) for a in range(len(rows)) for b in range(len(rows)))
    assert stats['global_MB'] > 0
    assert stats['pair_sessions'] == 3
    assert stats['pair_batches'] == 2
    assert stats['max_parallel_pairs'] <= 2
    assert stats['propagation_steps'] >= 1


def _garbled_case(case, rng):
    if case == 'exact':
        rows = [(('name', name), None) for name in [b'x', b'y', b'x', b'z', b'z']]
        return rows, [0, 0, 1, 1, 2], 1
    if case == 'transitive':
        images = [np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])]
        return [(('name', b'x'), image) for image in images], [0, 1, 2], 2 - 1
    if case == 'sym':                                 # symbols and embeddings mixed (EMNIST letters)
        e1 = ('emb', np.array([128, 0]), 0)
        e2 = ('emb', np.array([0, 128]), 0)
        rows = [(e1, None), (('sym', 'A'), None), (e1, None), (('sym', 'A'), None), (('sym', 'a'), None), (e2, None)]
        return rows, [0, 0, 1, 1, 2, 2], 1
    d = 16                                            # fuzzy: three planted classes, images 0/1
    base = rng.standard_normal((3, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    rows, owners = [], []
    for c in range(4):
        for g in rng.choice(3, 2, replace=False):
            e = base[g] + 0.05 * rng.standard_normal(d)
            e /= np.linalg.norm(e)
            img = np.zeros(8, np.int64)
            img[[g, g + 3, (g + int(rng.integers(2))) % 8]] = 1
            rows.append((('emb', np.round(e * 128).astype(np.int64), int(round(0.3 * (1 << 14)))), img))
            owners.append(c)
    return rows, owners, 2


@pytest.mark.skipif(not os.environ.get('MPSPDZ'), reason='set MPSPDZ for actual hegc/simhash integration')
@pytest.mark.parametrize('case', ['exact', 'transitive', 'fuzzy', 'sym'])
@pytest.mark.parametrize('pair', ['hegc', 'simhash'])
@pytest.mark.parametrize('group_protocol,gc', [('shamir', 'yao'), ('atlas', 'yao'), ('atlas', 'semi-bin')])
def test_garbled_pair_versions_match_ideal(case, pair, group_protocol, gc):
    from label_union.circuit_union import group
    from label_union import simhash
    rows, owners, t = _garbled_case(case, np.random.default_rng(3))
    match = (lambda a, b, tau: simhash.kw_match(a, b, tau, k=128)) if pair == 'simhash' else None
    expected, _ = group(rows, .10, t, owners, kw_match=match)
    got, stats, keys = runner().mpspdz_pairwise_group(rows, owners, t=t, m=max(owners.count(c) for c in set(owners)) + 1,
        timeout=120, pair_workers=2, group_version=2, group_protocol=group_protocol, pair_protocol=pair,
        simhash_bits=128, gc_protocol=gc)
    assert got == expected
    assert all((keys[a] == keys[b]) == (got[a] == got[b]) for a in range(len(rows)) for b in range(len(rows)))
    assert stats['pair_gc_MB'] > 0
    if case in ('fuzzy', 'sym'):
        assert len(set(expected)) < len(rows)        # the case exercises real matches
        if pair == 'hegc':
            assert stats['pair_he_MB'] > 0


def test_garbled_versions_need_grouping_v2():
    with pytest.raises(ValueError):
        runner().mpspdz_pairwise_group([(('name', b'x'), None)] * 3, [0, 1, 2], root='/nonexistent',
                                       pair_protocol='hegc', group_version=1)


def test_simhash_threshold_tracks_csls():
    from label_union import simhash
    rng = np.random.default_rng(0)
    d, agree, total = 64, 0, 0
    for _ in range(300):
        a = rng.standard_normal(d)
        a /= np.linalg.norm(a)
        b = a + rng.uniform(0.1, 1.5) * rng.standard_normal(d) / np.sqrt(d)
        b /= np.linalg.norm(b)
        ka = ('emb', np.round(a * 128).astype(np.int64), int(.45 * (1 << 14)))
        kb = ('emb', np.round(b * 128).astype(np.int64), int(.45 * (1 << 14)))
        exact = 2 * float(a @ b) - .9 >= .10
        if abs(2 * float(a @ b) - 1.0) > .15:          # away from the threshold
            agree += simhash.kw_match(ka, kb, .10, k=512) == exact
            total += 1
    assert total > 50 and agree / total > .95


def test_pca_projection_keeps_close_pairs_close():
    from label_union.pca import project
    rng = np.random.default_rng(5)
    anchors = rng.standard_normal((200, 32)) @ np.diag(np.r_[np.ones(8) * 5, np.ones(24) * .2])
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    e = anchors[0] + .01 * rng.standard_normal(32)
    rows = {'a': ('emb', np.round(anchors[0] * 128).astype(np.int64), 0),
            'b': ('emb', np.round(e / np.linalg.norm(e) * 128).astype(np.int64), 0), 's': ('sym', 'x')}
    out = project(rows, 8, anchors=anchors)
    assert out['s'] == rows['s'] and len(out['a'][1]) == 8
    assert out['a'][1] @ out['b'][1] > .95 * 128 * 128 and 0 < out['a'][2] <= 1 << 14


@pytest.mark.skipif(not os.environ.get('MPSPDZ'), reason='set MPSPDZ for the padding-size MPC')
@pytest.mark.parametrize('pair', ['semi', 'hegc'])
def test_padding_size_by_mpc(pair):
    from label_union.circuit_union import group
    rows, owners, t = _garbled_case('fuzzy', np.random.default_rng(3))
    rows, owners = rows + [rows[0]], owners + [0]            # client 0 holds the most rows (3)
    expected, _ = group(rows, .10, t, owners)
    got, stats, _ = runner().mpspdz_pairwise_group(rows, owners, t=t, timeout=120, pair_workers=2,
        group_version=2, group_protocol='atlas', pair_protocol=pair, pad_max='mpc')
    assert got == expected and stats['m'] == 3 and stats['pad_max_MB'] > 0
