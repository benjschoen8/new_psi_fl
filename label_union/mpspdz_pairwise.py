"""Setup-only hybrid: private two-party matches, then Shamir-shared grouping.

All processes run locally for measurement. Their private files are isolated and
removed after each invocation; the host itself is not a privacy boundary.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from label_union.mpspdz_group import _row_tokens, _h

PRIME = (1 << 127) - 1
# hemi (HE-based 2PC, matrix triples by BFV) needs an NTT-friendly prime, p = 1 mod 2^16
PRIME_NTT = 170141183460469231731687303715883253761
# hegc inner products: 61-bit NTT prime (= 1 mod 2^17); masked outputs s + rho < 2^59 never wrap
PRIME_HE = 1152921504614055937
SOURCES = Path(__file__).resolve().parent.parent / 'mpc'
# Pair versions: semi/hemi run pair_match.mpc (arithmetic 2PC, additive field shares of match bits).
# hegc: HE inner products (pair_inner.mpc on hemi) + garbled comparisons (pair_gc.mpc on yao);
# simhash: garbled SimHash test (pair_gc.mpc only). Both give XOR shares (grouping version 2).
PAIR_PROTOCOLS = ('semi', 'hemi', 'hegc', 'simhash')
GC_BINARIES = {'yao': 'yao-party.x', 'semi-bin': 'semi-bin-party.x'}   # binary 2PC of pair_gc.mpc
SYM_BITS = 8                        # garbled symbol test: character code of a one-letter keyword


def _sym_code(symbol):
    code = ord(symbol) if len(symbol) == 1 else -1
    if not 0 < code < 1 << SYM_BITS:
        raise ValueError('symbol keywords must be single 8-bit characters')
    return code
GC_L = 20                           # masked CSLS width in the garbled comparison
RHO = (1 << 18, 1 << 58)            # statistical masks of the HE outputs (|s| < 2^17)
_port_lock = threading.Lock()
_reserved = set()


class MPCSessionError(RuntimeError):
    """Public process diagnostics only; never includes raw private logs."""


def _pair_batches(n, cap):
    if n < 2 or cap < 1:
        raise ValueError('need at least two clients and a positive partner limit')
    ring = list(range(n)) + ([-1] if n % 2 else [])
    rounds = []
    for _ in range(len(ring) - 1):
        rounds.append([tuple(sorted((ring[i], ring[-1-i])))
                       for i in range(len(ring) // 2)
                       if -1 not in (ring[i], ring[-1-i])])
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return [sum(rounds[i:i+cap], []) for i in range(0, len(rounds), cap)]


def _reserve_ports(n):
    with _port_lock:
        for _ in range(200):
            base = random.randrange(20000, 60000 - n)
            ports = set(range(base, base + n))
            if ports & _reserved:
                continue
            sockets = []
            try:
                for port in ports:
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind(('127.0.0.1', port))
                _reserved.update(ports)
                return base
            except OSError:
                pass
            finally:
                for sock in sockets:
                    sock.close()
    raise RuntimeError('cannot reserve local MPC ports')


def _release_ports(base, n):
    with _port_lock:
        _reserved.difference_update(range(base, base + n))


def _private_write(path, values):
    with open(path, 'w', encoding='utf8', opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(' '.join(map(str, values)) + '\n')


def _parse_private(path, marker, count):
    values = {}
    try:
        with open(path, encoding='utf8') as stream:
            for line in stream:
                if not line.startswith(marker + ' '):
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError
                index, value = int(parts[1]), int(parts[2])
                if index in values or not 0 <= index < count or not -PRIME < value < PRIME:
                    raise ValueError
                values[index] = value
        if len(values) != count:
            raise ValueError
    except (OSError, ValueError) as error:
        # Never put private output values in an exception or benchmark report.
        raise RuntimeError('invalid or incomplete MPC private output') from None
    return [values[i] for i in range(count)]


def _parse_vector(path, marker, count):
    """One private output line '<marker> [v0, v1, ...]' with count integers."""
    try:
        found = re.search(rf'^{marker} \[([^\]]*)\]$', Path(path).read_text(), re.M)
        values = [int(v) for v in found.group(1).split(',')]
        if len(values) != count or not all(-PRIME < v < PRIME for v in values):
            raise ValueError
    except (OSError, ValueError, AttributeError):
        raise RuntimeError('invalid or incomplete MPC private output') from None
    return values


def _bridge_inputs(n, m, pairs, prefix):
    streams = []
    try:
        for p in range(n):
            path = Path(f'{prefix}-P{p}-0')
            streams.append(open(path, 'w', encoding='utf8',
                                opener=lambda p, flags: os.open(p, flags, 0o600)))
        for p, q in itertools.combinations(range(n), 2):
            for owner, path in zip((p, q), pairs[p, q]):
                shares = _parse_private(path, 'S', m * m)
                streams[owner].write(' '.join(map(str, shares)) + '\n')
    finally:
        for stream in streams:
            stream.close()


def _compile(home, source, args, edabit, timeout, flags=None):
    flags = flags or ['-F', '64', *(['-Y'] if edabit else [])]
    digest = hashlib.sha256(source.read_bytes() + repr((args, flags, sys.version)).encode())
    for path in [home / 'compile.py', *sorted((home / 'Compiler').rglob('*.py'))]:
        digest.update(path.read_bytes())
    stem = f'hybrid_{source.stem}_{digest.hexdigest()[:20]}'
    cache = home / 'Programs' / 'Schedules' / (stem + '.json')
    lock = home / 'Programs' / 'Source' / (stem + '.lock')
    start = time.perf_counter()
    with lock.open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved['files'] and all((home / p).is_file() for p in saved['files']):
                return saved['program'], time.perf_counter() - start, True
        (home / 'Programs' / 'Source' / (stem + '.mpc')).write_bytes(source.read_bytes())
        try:
            result = subprocess.run([sys.executable, 'compile.py', *flags, stem, *args],
                                    cwd=home, text=True, capture_output=True, timeout=timeout, check=True)
        except (subprocess.SubprocessError, OSError) as error:
            # Compiler diagnostics contain public source/parameters, not input files.
            diagnostic = ((getattr(error, 'stderr', '') or '') + (getattr(error, 'stdout', '') or ''))[-3000:]
            raise RuntimeError(f'{source.stem} compilation failed: {diagnostic}') from error
        match = re.search(r'Writing to .*?Programs/Schedules/(\S+)\.sch', result.stdout)
        if not match:
            raise RuntimeError('MPC compiler did not produce a schedule')
        program = match.group(1)
        files = re.findall(r'Writing to (Programs/\S+\.(?:bc|sch))', result.stdout)
        cache.write_text(json.dumps(dict(program=program, files=files)))
        return program, time.perf_counter() - start, False


def _run_session(*args, **kwargs):
    # Local start-up races (a port still bound, a refused connection) occasionally make a party
    # exit: rerun the same session on fresh ports; it recomputes the same outputs from the same
    # inputs. A persistent failure still raises after three attempts; timeouts/cancels never retry.
    for attempt in range(3):
        try:
            return _run_session_once(*args, **kwargs)
        except MPCSessionError as error:
            if attempt == 2 or 'timed out' in str(error) or 'cancelled' in str(error):
                raise


def _run_session_once(home, binary, n, program, input_prefix, output_prefix, log_prefix, timeout, *, cancel=None,
                      prime=PRIME):
    base = _reserve_ports(n)
    processes, logs = [], []
    start = time.monotonic()
    try:
        for p in range(n):
            if cancel is not None and cancel.is_set():
                raise MPCSessionError('MPC session cancelled after another pair failed')
            log = open(f'{log_prefix}-P{p}', 'w+', encoding='utf8',
                       opener=lambda p, flags: os.open(p, flags, 0o600))
            logs.append(log)
            name = Path(binary).name                   # binary circuits: no -P; yao: two parties, no -N
            processes.append(subprocess.Popen(
                [str(binary), *([] if name == 'yao-party.x' else ['-N', str(n)]), '-p', str(p), '-pn', str(base),
                 '-h', '127.0.0.1', *([] if name in GC_BINARIES.values() else ['-P', str(prime)]),
                 '-IF', str(input_prefix), '-OF', str(output_prefix), program],
                cwd=home, stdout=log, stderr=subprocess.STDOUT))
        while True:
            if cancel is not None and cancel.is_set():
                raise MPCSessionError('MPC session cancelled after another pair failed')
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                failures = []
                for p, code in enumerate(codes):
                    if code is not None and code != 0:
                        reason = f'signal={signal.Signals(-code).name}' if code < 0 else f'exit={code}'
                        failures.append(f'party={p} {reason} log_bytes={os.fstat(logs[p].fileno()).st_size}')
                        logs[p].seek(0)
                        diagnostic = logs[p].read()
                        for known in ('Address already in use', 'Cannot assign requested address',
                                      'Connection refused', 'certificate verify failed', 'bad_alloc',
                                      'Cannot allocate memory', 'Too many open files', 'Permission denied'):
                            if known in diagnostic:
                                failures.append(f'diagnostic={known}')
                hint = (' Check kernel/container OOM records; SIGKILL alone does not identify its cause.'
                        if -signal.SIGKILL in codes else '')
                raise MPCSessionError(f'MPC party failed in {Path(binary).name} stage={Path(log_prefix).name}: '
                                      + '; '.join(failures) + '; private logs were removed.' + hint)
            if all(code is not None for code in codes):
                break
            if time.monotonic() - start > timeout:
                raise MPCSessionError(f'MPC session timed out after {timeout:g} seconds in stage={Path(log_prefix).name}')
            time.sleep(.02)
        texts = []
        for log in logs:
            log.seek(0)
            texts.append(log.read())
        def number(pattern, text=texts[0]):
            found = re.search(pattern, text)
            return float(found.group(1)) if found else None
        sent = r'Data sent = ([\d.e+-]+) MB in ~\d+ rounds'           # each party: its own traffic and rounds
        return dict(wall_seconds=time.monotonic() - start,
                    global_MB=number(r'Global data sent = ([\d.e+-]+) MB'),
                    time_seconds=number(r'Time = ([\d.e+-]+) seconds'),
                    party_MB=[number(sent, t) or 0. for t in texts],
                    party_rounds=[number(r'Data sent = [\d.e+-]+ MB in ~(\d+) rounds', t) or 0. for t in texts])
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        _release_ports(base, n)


def _mpc_max(home, counts, protocol, prime, timeout):
    """Each client inputs its label count to an n-party MPC; returns (max, public stats)."""
    n = len(counts)
    program, compile_s, _ = _compile(home, SOURCES / 'label_max.mpc', [f'n={n}', 'bits=16'], False, timeout)
    (home / 'Player-Data').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='padmax-', dir=home / 'Player-Data') as directory:
        work = Path(directory)
        for p, count in enumerate(counts):
            _private_write(Path(f'{work}/in-P{p}-0'), [count])
        stats = _run_session(home, home / f'{protocol}-party.x', n, program, work / 'in', work / 'out',
                             work / 'log', timeout, prime=prime)
        found = re.search(r'^M (\d+)$', Path(f'{work}/out-P0-0').read_text(), re.M)
    if not found:
        raise RuntimeError('invalid or incomplete MPC output')
    stats.update(compile_seconds=compile_s)
    return int(found.group(1)), stats


def mpspdz_pairwise_group(rows, owners, tau=.10, t=2, m=None, root=None, fix=7,
                         timeout=600, pair_concurrency=2, pair_workers=4,
                         edabit=True, prefix='parallel', block_rows=64, group_edabit=False,
                         group_version=1, group_protocol='shamir', pair_protocol='semi',
                         simhash_bits=256, simhash_u0=1.0, gc_protocol='yao', pad_max='plain'):
    if (not rows or len(rows) != len(owners) or any(not isinstance(p, int) or p < 0 for p in owners)
            or fix != 7 or prefix not in ('serial', 'parallel')
            or not isinstance(block_rows, int) or block_rows < 1
            or group_version not in (1, 2) or group_protocol not in ('shamir', 'atlas')
            or pair_protocol not in PAIR_PROTOCOLS
            or (pair_protocol in ('hegc', 'simhash') and group_version != 2) or gc_protocol not in GC_BINARIES
            or pad_max not in ('plain', 'mpc')
            or not isinstance(simhash_bits, int) or simhash_bits < 64 or simhash_bits % 64
            or pair_concurrency < 1 or pair_workers < 1 or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('invalid hybrid MPC inputs or options')
    n = max(owners) + 1
    if n < 3 or set(owners) != set(range(n)):
        raise ValueError('global Shamir grouping requires at least three nonempty clients')
    per = [[i for i, owner in enumerate(owners) if owner == p] for p in range(n)]
    if m is None and pad_max == 'plain':
        m = max(map(len, per))
    if m is not None and (not isinstance(m, int) or m < max(map(len, per))):
        raise ValueError('padding must accommodate every client')
    fuzzy = all(kw[0] != 'name' for kw, _ in rows)
    if not fuzzy and any(kw[0] != 'name' for kw, _ in rows):
        raise ValueError('cannot mix exact and fuzzy inputs')
    sym = int(fuzzy and any(kw[0] == 'sym' for kw, _ in rows))
    embeddings = [kw[1] for kw, _ in rows if kw[0] == 'emb']
    d = len(embeddings[0]) if embeddings else 1
    nimg = len(rows[0][1]) if rows[0][1] is not None else 0
    if any(len(e) != d for e in embeddings) or any((0 if img is None else len(img)) != nimg for _, img in rows):
        raise ValueError('all rows must use the same feature dimensions')
    garbled = pair_protocol in ('hegc', 'simhash')
    if garbled and nimg > 64:
        raise ValueError('garbled pair versions support at most 64 image anchors')
    kw_mode = ('eq' if not fuzzy else 'masked' if pair_protocol == 'hegc' else 'simhash') if garbled else None
    home = Path(root or os.environ.get('MPSPDZ', '')).expanduser().resolve()
    prime = PRIME_NTT if pair_protocol == 'hemi' else PRIME     # both stages share one field
    pair_binaries = ([GC_BINARIES[gc_protocol]] + (['hemi-party.x'] if kw_mode == 'masked' else [])
                     if garbled else [f'{pair_protocol}-party.x'])
    for name in ('compile.py', *pair_binaries, f'{group_protocol}-party.x'):
        if not (home / name).is_file():
            raise RuntimeError(f'MP-SPDZ hybrid backend requires {name}; set MPSPDZ to a complete installation')
    pad_stats = None
    if m is None:                       # pad_max='mpc': only the largest label count is opened
        m, pad_stats = _mpc_max(home, [len(own) for own in per], group_protocol, prime, timeout)
    args = [f'm={m}', f'd={d if fuzzy else 0}', f'nimg={nimg}', f'tau={round(tau * (1 << (2 * fix)))}',
            f't={t}', f"mode={'fuzzy' if fuzzy else 'exact'}", f'sym={sym}']
    tau_i = round(tau * (1 << (2 * fix)))
    inner_program = None
    if not garbled:
        pair_program, pair_compile, pair_cached = _compile(home, SOURCES / 'pair_match.mpc', args, edabit, timeout)
    else:
        from label_union import simhash as sh
        C = sh.threshold(tau, simhash_bits, sh.SCALE, simhash_u0)[1]
        pair_program, pair_compile, pair_cached = _compile(home, SOURCES / 'pair_gc.mpc',
            [f'm={m}', f'kw={kw_mode}', f'nimg={nimg}', f't={t}', f'L={GC_L}', f'k={simhash_bits}',
             f'S={sh.SCALE}', f'C={C}', f'G={sh.G_BITS}', f'sym={sym}', f'sb={SYM_BITS}'], False, timeout,
            flags=(['-G'] if gc_protocol == 'yao' else []) + ['-B', '64'])
        if kw_mode == 'masked':
            inner_program, inner_compile, inner_cached = _compile(home, SOURCES / 'pair_inner.mpc',
                [f'm={m}', f'd={d}'], False, timeout, flags=['-F', '40'])
            pair_compile += inner_compile
            pair_cached = pair_cached and inner_cached
    group_program, group_compile, group_cached = _compile(home, SOURCES / ('shared_graph_group.mpc' if group_version == 1 else
                                                                  f'shared_graph_group_v{group_version}.mpc'),
        [f'n={n}', f'm={m}', f'prefix={prefix}', f'block={block_rows}', *(['share=xor'] if garbled else [])],
        group_edabit, timeout)
    batches = _pair_batches(n, pair_concurrency)
    start = time.perf_counter()
    tokens = []
    for own in per:
        padded = [rows[own[i]] if i < len(own) else (None, None) for i in range(m)]
        if not garbled:
            tokens.append([token for row in padded for token in _row_tokens(*row, fuzzy, sym, d, nimg)])
            continue
        gc = [int(kw is not None) for kw, _ in padded]                    # pair_gc.mpc input order
        emb = lambda kw: kw is not None and kw[0] == 'emb'
        if sym:
            gc += [int(kw is not None and kw[0] == 'sym') for kw, _ in padded]
            gc += [_sym_code(kw[1]) if kw is not None and kw[0] == 'sym' else 0 for kw, _ in padded]
        if kw_mode == 'eq':
            gc += [_h(kw[1]) if kw is not None else 0 for kw, _ in padded]
        elif kw_mode == 'simhash':
            gc += [w for kw, _ in padded for w in sh.words(sh.code(kw[1], simhash_bits) if emb(kw)
                                                            else [0] * simhash_bits)]
            gc += [sh.g_share(kw[2], simhash_bits, sh.SCALE, simhash_u0) if emb(kw) else 0 for kw, _ in padded]
        tail = [sum(int(b) << i for i, b in enumerate(img)) if img is not None else 0
                for _, img in padded] if nimg else []          # after the masked scores, if any
        he = ([v for kw, _ in padded for v in ([int(x) for x in kw[1]] + [int(kw[2])] if emb(kw)
                                                else [0] * (d + 1))] if kw_mode == 'masked' else None)
        tokens.append((gc, he, tail))
    private_root = home / 'Player-Data'
    private_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hybrid-', dir=private_root) as directory:
        work = Path(directory)
        lock = threading.Lock()
        cancel = threading.Event()
        active = peak = 0
        pair_outputs, pair_stats, pair_party = {}, [], {}

        def matching(pair):
            nonlocal active, peak
            p, q = pair
            stem = work / f'pair-{p}-{q}'
            inp, out = Path(f'{stem}-Input'), Path(f'{stem}-Output')
            for party, owner in enumerate((p, q)):
                if not garbled:
                    _private_write(Path(f'{inp}-P{party}-0'), tokens[owner])
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if garbled:
                    return pair, *garbled_matching(p, q, stem)
                stats = _run_session(home, home / f'{pair_protocol}-party.x', 2, pair_program, inp, out,
                                     Path(f'{stem}-Log'), timeout, cancel=cancel, prime=prime)
                paths = [Path(f'{out}-P{party}-0') for party in range(2)]
                for path in paths:
                    _parse_private(path, 'S', m * m)
                return pair, paths, stats
            finally:
                with lock:
                    active -= 1

        def garbled_matching(p, q, stem):
            # Each party draws its own masks; z_e: fresh XOR mask bit of each party per entry.
            rand, entries = secrets.SystemRandom(), m * m
            z = [[rand.getrandbits(1) for _ in range(entries)] for _ in range(2)]
            gc = [list(tokens[p][0]), list(tokens[q][0])]
            he_stats = None
            if kw_mode == 'masked':
                rho = [rand.randrange(*RHO) for _ in range(entries)]
                inp, out = Path(f'{stem}-HE-Input'), Path(f'{stem}-HE-Output')
                _private_write(Path(f'{inp}-P0-0'), tokens[p][1])
                _private_write(Path(f'{inp}-P1-0'), tokens[q][1] + rho)
                he_stats = _run_session(home, home / 'hemi-party.x', 2, inner_program, inp, out,
                                        Path(f'{stem}-HE-Log'), timeout, cancel=cancel, prime=PRIME_HE)
                x = _parse_vector(Path(f'{out}-P0-0'), 'X', entries)
                gc[0] += [v % (1 << GC_L) for v in x]
                gc[1] += [(r + tau_i) % (1 << GC_L) for r in rho]
            gc = [gc[0] + tokens[p][2], gc[1] + tokens[q][2]]
            inp, out = Path(f'{stem}-Input'), Path(f'{stem}-Output')
            for party in range(2):
                _private_write(Path(f'{inp}-P{party}-0'), gc[party] + z[party])
            stats = _run_session(home, home / GC_BINARIES[gc_protocol], 2, pair_program, inp, out,
                                 Path(f'{stem}-Log'), timeout, cancel=cancel)
            found = re.search(r'^W (-?\d+)$', Path(f'{out}-P0-0').read_text(), re.M)
            if not found:
                raise RuntimeError('invalid or incomplete MPC private output')
            w = int(found.group(1)) % (1 << entries)       # printed as a signed entries-bit integer
            shares = [[((w >> e) & 1) ^ z[0][e] for e in range(entries)], z[1]]
            paths = [Path(f'{stem}-Share-P{party}') for party in range(2)]
            for path, share in zip(paths, shares):
                with open(path, 'w', encoding='utf8', opener=lambda f, flags: os.open(f, flags, 0o600)) as stream:
                    stream.writelines(f'S {e} {v}\n' for e, v in enumerate(share))
            stats['gc_MB'] = stats['global_MB']
            if he_stats is not None:
                for key in ('party_MB', 'party_rounds'):                  # both sessions of the pair
                    stats[key] = [a + b for a, b in zip(stats[key], he_stats[key])]
                stats['he_MB'] = he_stats['global_MB']
                stats['wall_seconds'] += he_stats['wall_seconds']
                stats['global_MB'] = (None if None in (stats['global_MB'], he_stats['global_MB'])
                                      else stats['global_MB'] + he_stats['global_MB'])
            return paths, stats

        matching_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=pair_workers) as pool:
            for batch in batches:
                futures = [pool.submit(matching, pair) for pair in batch]
                try:
                    for future in as_completed(futures):
                        pair, paths, stats = future.result()
                        pair_outputs[pair] = paths
                        pair_stats.append(stats)
                        pair_party[pair] = stats
                except BaseException:
                    cancel.set()
                    for future in futures:
                        future.cancel()
                    raise
        matching_seconds = time.perf_counter() - matching_start
        bridge_start = time.perf_counter()
        inp, out = work / 'graph-Input', work / 'graph-Output'
        _bridge_inputs(n, m, pair_outputs, inp)
        bridge_seconds = time.perf_counter() - bridge_start
        graph = _run_session(home, home / f'{group_protocol}-party.x', n, group_program, inp, out, work / 'graph-Log', timeout,
                             prime=prime)
        keys = [None] * len(rows)
        for p in range(n):
            values = _parse_private(Path(f'{out}-P{p}-0'), 'K', m)
            for i, row in enumerate(per[p]):
                keys[row] = values[i]
        output0 = Path(f'{out}-P0-0').read_text()
        steps = re.search(r'propagation steps \(incl\. the public first step\): (\d+)', output0)
    first = {}
    roots = [first.setdefault(key, i) for i, key in enumerate(keys)]
    # per client: own traffic sent in every session it is a party of, traffic received from the other
    # party (pairs) or an even part of the others' traffic (n-party sessions), and rounds (summed over its
    # sessions; pair sessions divided by pair_concurrency, the partners a client runs at once)
    sent, recv, rounds, pair_rounds = [0.] * n, [0.] * n, [0.] * n, [0.] * n
    for (p, q), st in pair_party.items():
        for side, (me, other) in enumerate(((p, 1), (q, 0))):
            sent[me] += st['party_MB'][side]
            recv[me] += st['party_MB'][other]
            rounds[me] += st['party_rounds'][side] / pair_concurrency    # its pairs run concurrently
            pair_rounds[me] += st['party_rounds'][side]                  # undivided (setup_smoke_hybrid)
    for st in (graph, pad_stats):
        if st is None:
            continue
        total = sum(st['party_MB'])
        for c in range(n):
            sent[c] += st['party_MB'][c]
            recv[c] += (total - st['party_MB'][c]) / (n - 1)
            rounds[c] += st['party_rounds'][c]
    volumes = [stats['global_MB'] for stats in pair_stats]
    pair_mb = sum(volumes) if all(value is not None for value in volumes) else None
    total_mb = pair_mb + graph['global_MB'] if pair_mb is not None and graph['global_MB'] is not None else None
    stats = dict(backend='pairwise', client_sent_MB=sent, client_received_MB=recv, client_rounds=rounds,
                 client_pair_rounds=pair_rounds, n=n, m=m, rows=n*m, d=d if fuzzy else 0, nimg=nimg,
                 mode='fuzzy' if fuzzy else 'exact', sym=sym, prefix=prefix, block_rows=min(n*m, block_rows), edabit=edabit,
                 group_edabit=group_edabit, group_version=group_version, group_protocol=group_protocol,
                 pad_max=pad_max, pad_max_MB=pad_stats and pad_stats['global_MB'],
                 pad_max_seconds=pad_stats and pad_stats['wall_seconds'],
                 pair_protocol=pair_protocol, pair_kw=kw_mode, gc_protocol=gc_protocol if garbled else None,
                 simhash_bits=simhash_bits if kw_mode == 'simhash' else None,
                 simhash_u0=simhash_u0 if kw_mode == 'simhash' else None,
                 pair_he_MB=sum(st.get('he_MB') or 0 for st in pair_stats) if kw_mode == 'masked' else None,
                 pair_gc_MB=sum(st.get('gc_MB') or 0 for st in pair_stats) if garbled else None,
                 field_prime=str(prime), compile_seconds=pair_compile + group_compile,
                 pair_compile_seconds=pair_compile, group_compile_seconds=group_compile,
                 compile_cache_hits=int(pair_cached) + int(group_cached), wall_seconds=time.perf_counter()-start,
                 matching_seconds=matching_seconds, bridge_seconds=bridge_seconds, group_seconds=graph['wall_seconds'],
                 pair_sessions=len(pair_stats), pair_batches=len(batches), pair_concurrency=pair_concurrency,
                 pair_workers=pair_workers, max_parallel_pairs=peak, pair_global_MB=pair_mb,
                 group_global_MB=graph['global_MB'], global_MB=total_mb, rounds=None,
                 propagation_steps=int(steps.group(1)) if steps else None)
    return roots, stats, keys
