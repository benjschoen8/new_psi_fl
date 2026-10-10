"""Setup-only no-cluster benchmark; no models or training are constructed.

    python -m setup_smoke_hybrid
    python -m setup_smoke_hybrid --clients 3 5 --repeats 3 --out setup_results_hybrid
    python -m setup_smoke_hybrid --data synthetic --clients 3 --labels 3
    MPSPDZ=/path/to/mp-spdz python -m setup_smoke_hybrid --methods plain exact fuzzy

Setup means image signatures/anchors, label grouping, bucket union, and public-key
distribution. Model initialization, data loading, training, and evaluation are excluded.
Fuzzy uses the existing text encoder (cache or optional sentence-transformers).
Default inputs are MNIST, EMNIST byclass, and CIFAR-10 through the original dataset
loader, transforms, partitions, label metadata, and per-label image sampler.

On x86-64 Linux a prebuilt MP-SPDZ binary is downloaded. On other platforms (ARM,
macOS, ...) MP-SPDZ is cloned and compiled from source into ~/.cache/mp-spdz
(override with MPSPDZ_BUILD_DIR; pin a tag/branch with MPSPDZ_REF).
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from label_union.circuit_union_hybrid import circuit_union_with_keys
from label_union.domain import anchor_matrix, client_sets
from label_union.mpspdz_pairwise import MPCSessionError

from setup_smoke import ensure_mpspdz, ensure_certificates, save_report, fixture, real_inputs


def ensure_pairwise_backend(root, names=('semi-party.x',)):
    """Reuse/build the hybrid executables (default: semi-honest two-party) in the selected installation."""
    for name in names:
        _ensure_binary(root, name)


def _ensure_binary(root, name):
    binary = root / name
    if binary.is_file() and os.access(binary, os.X_OK):
        return
    if not (root / 'Makefile').is_file():
        raise RuntimeError(f'MP-SPDZ at {root} has no {name} or source Makefile. '
                           f'Set MPSPDZ to a source installation with {name} built.')
    jobs = os.environ.get('MPSPDZ_JOBS', '4')
    try:
        if int(jobs) < 1:
            raise ValueError
    except ValueError as error:
        raise RuntimeError('MPSPDZ_JOBS must be a positive integer') from error
    print(f'Compiling MP-SPDZ {name} for the hybrid setup...', file=sys.stderr)
    try:
        subprocess.run(['make', '-j', jobs, name], cwd=root,
                       check=True, stdout=sys.stderr)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f'Building {name} in {root} failed (see output above). '
                           'Check native MP-SPDZ build dependencies and run make setup in that directory; '
                           'pairwise setup requires the real executable.') from error
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f'MP-SPDZ build did not produce executable {binary}')


def benchmark(method, labels, samples, bucket_bits=16, workers=1, keywords=None, *,
              mpc_mode='pairwise', pair_concurrency=2, pair_workers=4, mpc_timeout=600,
              group_prefix='parallel', group_block_rows=64, group_edabit=False,
              group_version=1, group_protocol='shamir', pair_protocol='semi',
              pca_dim=None, simhash_bits=256, simhash_u0=1.0, gc_protocol='yao', pad_max='plain', mpc_model=False,
              pad_to=None):
    # Make every trial include cold public image-anchor construction. The single
    # process then shares that cache, as in the current simulation, not n hosts.
    anchor_matrix.cache_clear()
    start = time.perf_counter()
    sets = [client_sets(s, names) for s, names in zip(samples, labels)]
    image_seconds = time.perf_counter() - start
    union_start = time.perf_counter()
    options = {}
    pairwise = mpc_mode == 'pairwise' and method != 'plain'
    if method != 'plain':
        mpc_options = {}
        if pairwise:
            mpc_options.update(prefix=group_prefix, timeout=mpc_timeout, block_rows=group_block_rows,
                               pair_concurrency=pair_concurrency, pair_workers=pair_workers,
                               group_edabit=group_edabit, group_version=group_version,
                               group_protocol=group_protocol, pair_protocol=pair_protocol)
            mpc_options.update(pad_max='plain' if pad_to else pad_max)   # public padding policy: no padding MPC
            if mpc_model:
                mpc_options.update(model=True)
            if pair_protocol in ('hegc', 'simhash'):
                mpc_options.update(gc_protocol=gc_protocol)
            if pair_protocol == 'simhash':
                mpc_options.update(simhash_bits=simhash_bits, simhash_u0=simhash_u0)
            if pca_dim:
                mpc_options.update(pca_dim=pca_dim)
        options = dict(mpc_backend=mpc_mode, mpc_options=mpc_options)
    index, _, _, U, stats = circuit_union_with_keys(
        labels, keywords if method == 'fuzzy' and keywords is not None else [{x: x for x in own} for own in labels], sets,
        fuzzy=method == 'fuzzy', secure=method != 'plain',
        bucket_bits=bucket_bits, workers=workers, m=pad_to if pairwise else None, **options)
    elapsed = time.perf_counter() - start
    quality = union_quality(labels, index, U) if index else {}
    union_seconds = time.perf_counter() - union_start
    mpc = stats.get('mpc', {})
    measured = mpc.get('measured')
    if pairwise and not measured:
        raise RuntimeError('Pairwise setup did not return real MP-SPDZ measurements')
    upload = stats['setup_upload_bytes_per_client']
    download = stats['setup_download_bytes_per_client']
    return dict(
        method=method, mpc_mode=mpc_mode, group_prefix=group_prefix if pairwise else 'serial',
        clients=len(labels), labels_per_client=max(map(len, labels)), padded_to=pad_to,
        labels_per_client_min=min(map(len, labels)), labels_per_client_mean=float(np.mean(list(map(len, labels)))),
        bucket_bits=bucket_bits, union_size=U,
        backend=('plain' if method == 'plain' else
                 'mpc-model' if pairwise and mpc_model else
                 'mp-spdz-pairwise' if pairwise else
                 'mp-spdz' if measured else 'ideal-functionality'),
        setup_wall_seconds=elapsed, image_setup_seconds=image_seconds,
        union_wall_seconds=union_seconds,
        mpc_compile_seconds=(measured or {}).get('compile_seconds'),
        mpc_wall_seconds=(measured or {}).get('wall_seconds'),
        mpc_global_MB=(measured or {}).get('global_MB'),
        matching_seconds=(measured or {}).get('matching_seconds'),
        group_seconds=(measured or {}).get('group_seconds'),
        bridge_seconds=(measured or {}).get('bridge_seconds'),
        matching_compile_seconds=(measured or {}).get('pair_compile_seconds'),
        group_compile_seconds=(measured or {}).get('group_compile_seconds'),
        compile_cache_hits=(measured or {}).get('compile_cache_hits'),
        group_block_rows=(measured or {}).get('block_rows'),
        group_edabit=(measured or {}).get('group_edabit'),
        group_version=(measured or {}).get('group_version'),
        group_protocol=(measured or {}).get('group_protocol'),
        pair_protocol=(measured or {}).get('pair_protocol'),
        pca_dim=(measured or {}).get('pca_dim'),
        simhash_bits=(measured or {}).get('simhash_bits'),
        pair_he_MB=(measured or {}).get('pair_he_MB'),
        pair_gc_MB=(measured or {}).get('pair_gc_MB'),
        gc_protocol=(measured or {}).get('gc_protocol'),
        pad_max=(measured or {}).get('pad_max'),
        pad_max_MB=(measured or {}).get('pad_max_MB'),
        pair_batches=(measured or {}).get('pair_batches'),
        pair_sessions=(measured or {}).get('pair_sessions'),
        max_parallel_pairs=(measured or {}).get('max_parallel_pairs'),
        estimated_upload_bytes_per_client=upload,
        estimated_download_bytes_per_client=download,
        estimated_upload_bytes_total=None if upload is None else len(labels) * upload,
        estimated_download_bytes_total=None if download is None else len(labels) * download,
        **quality, secagg_accounted_bytes=stats.get('secagg'), mpc_measured=measured,
        protocol_stats=stats)


def union_quality(labels, index, U):
    """Experimenter only: the union against the true label names (same name = same class; STL-10 uses
    CIFAR-10's names). pair MCC over all pairs of (client, label) rows of different clients: positive =
    same true name, predicted positive = same union index."""
    from label_union import index_metrics
    rows = [(c, x) for c, own in enumerate(labels) for x in own]
    tp = fp = fn = tn = 0
    for i, (c, x) in enumerate(rows):
        for d, y in rows[i + 1:]:
            if c == d:
                continue
            same, pred = x == y, index[c][x] == index[d][y]
            tp, fp, fn, tn = tp + (same and pred), fp + (pred and not same), fn + (same and not pred), tn + (not same and not pred)
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    m = index_metrics(labels, index, U, sorted({x for own in labels for x in own}))
    names_of, slots_of = {}, {}                                   # experimenter diagnostics: what went wrong
    for c, own in enumerate(labels):
        for x in own:
            names_of.setdefault(index[c][x], set()).add(x)
            slots_of.setdefault(x, set()).add(index[c][x])
    merged = sorted((sorted(v) for v in names_of.values() if len(v) > 1), key=len, reverse=True)
    split = sorted((x, len(v)) for x, v in slots_of.items() if len(v) > 1)
    return dict(union_exact=bool(m['exact']), pair_mcc=(tp * tn - fp * fn) / den if den else float(fp == fn == 0),
                pair_tp=tp, pair_fp=fp, pair_fn=fn, split_labels=len(m['split_labels']), merged_indices=len(m['merged_indices']),
                merged_groups=merged[:60], split_names=split[:60])


UNION_ROUNDS = 3        # union/key SecAgg after the MPC: upload tags, masked keys, download result


def setup_comm(row, mbps, rtt_ms, pair_workers=1, per_client=False):
    """Per-client setup communication: measured MP-SPDZ traffic (each party's own 'Data sent', plus what
    it receives) and rounds, plus the union/key SecAgg bytes; modelled time = bytes * 8 / bandwidth +
    rounds * RTT on the slowest client (MPC rounds summed over its sessions: an upper bound)."""
    mpc, sa, n = row.get('mpc_measured') or {}, row.get('secagg_accounted_bytes') or {}, row['clients']
    union_up = (sa.get('payload_up', 0) + sa.get('control_up', 0)) / n
    union_down = sa.get('control_down', 0) / n
    sent = mpc.get('client_sent_MB') or [0.] * n
    recv = mpc.get('client_received_MB') or [0.] * n
    rounds = mpc.get('client_rounds') or [0.] * n
    total = [(s + r) * 1e6 + union_up + union_down for s, r in zip(sent, recv)]
    secs = [b * 8 / (mbps * 1e6) + (k + UNION_ROUNDS) * rtt_ms / 1e3 for b, k in zip(total, rounds)]
    # deployment estimate: each client is its own host, disjoint pairs run at once (round-robin schedule:
    # n - 1 rounds, n odd: n), so a client's latency = rounds x one pair session (measured: the host's
    # matching time x workers / sessions; exact with --pair-workers 1, no contention) + grouping +
    # padding MPC + modelled network time
    sessions = mpc.get('pair_sessions') or 0
    pair_s = mpc.get('matching_seconds', 0) * min(pair_workers, sessions) / sessions if sessions else 0.
    sched = (n - 1 + n % 2) if sessions else 0
    deploy = dict(pair_workers=pair_workers, per_client_threads=per_client, host_compute_seconds=(mpc.get('matching_seconds') or 0.)
                  + (mpc.get('group_seconds') or 0.) + (mpc.get('pad_max_seconds') or 0.), pair_session_seconds=pair_s, deploy_pair_rounds=sched,
                  deploy_matching_seconds=sched * pair_s, deploy_group_seconds=mpc.get('group_seconds') or 0.,
                  deploy_pad_seconds=mpc.get('pad_max_seconds') or 0.)
    deploy['deploy_compute_seconds'] = (deploy['deploy_matching_seconds'] + deploy['deploy_group_seconds']
                                        + deploy['deploy_pad_seconds'])
    deploy['deploy_seconds'] = deploy['deploy_compute_seconds'] + max(secs)
    # network time of the slowest client when it runs c of its pair sessions at once: bandwidth-bound
    # part unchanged, the pair sessions' round trips overlap c-fold (grouping rounds do not)
    raw = mpc.get('client_pair_rounds') or [0.] * n
    other = [k - r / max(1, mpc.get('pair_concurrency') or 2) for k, r in zip(rounds, raw)]   # grouping + padding
    net = lambda c: max(b * 8 / (mbps * 1e6) + (r / c + o + UNION_ROUNDS) * rtt_ms / 1e3
                        for b, r, o in zip(total, raw, other))
    deploy.update(comm_seconds_sequential=net(1),          # per client: each client in one session at a time
                  comm_seconds_workers=net(1 if per_client else max(1, min(pair_workers, n - 1))))
    return dict(**deploy, comm_MB_per_client_mean=sum(total) / n / 1e6, comm_MB_per_client_max=max(total) / 1e6,
                mpc_MB_per_client_mean=sum(s + r for s, r in zip(sent, recv)) / n,
                union_MB_per_client=(union_up + union_down) / 1e6,
                comm_rounds_per_client_max=max(rounds) + UNION_ROUNDS,
                comm_seconds_model_max=max(secs), comm_seconds_model_mean=sum(secs) / n,
                net_mbps=mbps, net_rtt_ms=rtt_ms)


def approx_check(labels, samples, keywords, bits=(128, 256, 512), dims=(32, 48, 64, 128), tau=None, t=2):
    """Plaintext: exact CSLS grouping vs simhash / PCA groupings of the same fuzzy rows."""
    from label_union import circuit_union as cu, simhash
    from label_union.pca import project
    if keywords is None:
        raise ValueError('--approx-check needs real data (fuzzy keywords)')
    tau = cu.TAU if tau is None else tau
    sets = [client_sets(s, names) for s, names in zip(samples, labels)]
    kws = [cu.keyword_rows(own, True) for own in keywords]

    def rows_of(kw_rows):
        rows, owners = [], []
        for c, own in enumerate(labels):
            imgs = cu.image_rows(sets[c])
            for x in own:
                if kw_rows[c][x][0] == 'emb':         # symbols: not in this comparison
                    rows.append((kw_rows[c][x], imgs[x]))
                    owners.append(c)
        return rows, owners
    rows, owners = rows_of(kws)
    exact, _ = cu.group(rows, tau, t, owners)
    out = []
    for k in bits:
        approx, _ = cu.group(rows, tau, t, owners, kw_match=lambda a, b, tau: simhash.kw_match(a, b, tau, k=k))
        out.append(dict(approx='simhash', bits=k, **simhash.agreement(exact, approx)))
    for k in dims:
        prow, _ = rows_of([project(own, k) for own in kws])
        approx, _ = cu.group(prow, tau, t, owners)
        out.append(dict(approx='pca', dim=k, **simhash.agreement(exact, approx)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clients', type=int, nargs='+', default=[50],
                        help='client counts to benchmark sequentially (default: 50, the main split; includes the special clients)')
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real')
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--exp-conf', type=Path, default=Path('config.yaml'))
    parser.add_argument('--samples-per-label', type=int, default=16)
    parser.add_argument('--class-subsets', default='5,6',
                        help='same as training CLI: LO,HI or even (default 5,6: the main experiment\'s label split); '
                             'none: the original Dirichlet partition')
    parser.add_argument('--class-share', choices=['split', 'full'], default='full',
                        help='default full: every holder gets all images of its classes (the main experiment)')
    parser.add_argument('--num-train-cifar10stl10', type=int, default=2,
                        help='special clients with CIFAR-10 + STL-10 merged per class name, on top of --clients '
                             '(default 2, the main experiment)')
    parser.add_argument('--noniid-partition', default='dirichlet',
                        choices=['dirichlet', 'noniid_label', 'quantity_skew', 'quantity_skew_equalSize'])
    parser.add_argument('--fuzzy-langs', default='en0,en1', help='same client keyword writers as training CLI')
    parser.add_argument('--labels', type=int, nargs='+', help='synthetic only: labels per client (default 3)')
    parser.add_argument('--methods', nargs='+', choices=['plain', 'exact', 'fuzzy', 'similar'], default=['fuzzy'],
                        help='similar = new name of fuzzy (same method)')
    parser.add_argument('--bucket-bits', type=int,
                        help='default 20 for real data (production size), 16 for synthetic')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--mpc-mode', choices=['global', 'pairwise', 'both'], default='pairwise',
                        help='original global MPC, new hybrid (default), or compare both')
    parser.add_argument('--pair-concurrency', type=int, default=2,
                        help='maximum active matching sessions per client (default 2)')
    parser.add_argument('--pair-workers', type=int, default=4,
                        help='maximum simultaneous matching sessions on this host (default 4); 0 = one per client: '
                             'every client runs its sessions one at a time, all clients at once (round-robin rounds '
                             'of n/2 disjoint pairs, n MPC processes), i.e. a per-client deployment on this host')
    parser.add_argument('--pad-to', type=int, default=None,
                        help='public padding policy: every client pads to this many labels (no padding MPC); '
                             'must be >= every client\'s label count')
    parser.add_argument('--mpc-timeout', type=float, default=600,
                        help='timeout per hybrid MPC session/compilation in seconds; original global backend unchanged (default 600)')
    parser.add_argument('--group-prefix', choices=['serial', 'parallel'], default='parallel',
                        help='new hybrid grouping prefix; global comparison retains original serial prefix')
    parser.add_argument('--group-block-rows', type=int, default=64,
                        help='hybrid grouping workspace rows (default 64); smaller uses less memory but more rounds')
    parser.add_argument('--group-edabit', action='store_true',
                        help='enable mixed edaBit grouping for comparison; default arithmetic-only grouping')
    parser.add_argument('--group-version', type=int, choices=[1, 2],
                        help='hybrid grouping circuit: 1 original, 2 comparison-free convergence check (new version); '
                             'default 1, or 2 for the garbled pair versions (required)')
    parser.add_argument('--group-protocol', choices=['shamir', 'atlas'], default='shamir',
                        help='honest-majority grouping protocol (atlas: less traffic for many parties)')
    parser.add_argument('--pair-protocol', choices=['semi', 'hemi', 'hegc', 'simhash'], default='semi',
                        help='two-party matching: semi (OT) or hemi (HE matrix triples; NTT prime for both stages), '
                             'hegc (HE inner products + garbled comparison, exact CSLS) or simhash (garbled SimHash test, '
                             'approximate CSLS); hegc/simhash use grouping version 2')
    parser.add_argument('--gc-protocol', choices=['yao', 'semi-bin'], default='yao',
                        help='hegc/simhash binary 2PC: yao (garbled circuit) or semi-bin (OT-based GMW, less traffic, more rounds)')
    parser.add_argument('--pad-max', choices=['plain', 'mpc'], default='plain',
                        help='padding size m = largest label count: computed in plaintext (default) or by an n-party '
                             'MPC that opens only the maximum')
    parser.add_argument('--pca-dim', type=int,
                        help='project keyword embeddings onto the top-k PCA axes of the public anchor words (fuzzy only)')
    parser.add_argument('--simhash-bits', type=int, default=256, help='simhash: hyperplanes, multiple of 64 (default 256)')
    parser.add_argument('--simhash-u0', type=float, default=1.0,
                        help='simhash: linearisation point of arccos((tau + r + r\')/2) (default 1.0)')
    parser.add_argument('--datasets', default='MNIST,EMNIST,CIFAR10,FashionMNIST,STL10',
                        help='real data: datasets, clients split evenly in this order (default: the five of the '
                             'main experiment; the earlier three: MNIST,EMNIST,CIFAR10)')
    parser.add_argument('--mpc-model', action='store_true',
                        help='no MP-SPDZ: grouping as its ideal functionality, MPC traffic / rounds / time from the '
                             'fitted cost model (label_union.mpc_model; hegc, atlas, grouping version 2)')
    parser.add_argument('--net-mbps', type=float, default=100., help='modelled link speed per client (Mbit/s)')
    parser.add_argument('--net-rtt-ms', type=float, default=20., help='modelled round-trip time (ms)')
    parser.add_argument('--approx-check', action='store_true',
                        help='no MPC: compare exact CSLS groupings with simhash and PCA groupings on the same fuzzy inputs')
    parser.add_argument('--approx-bits', type=int, nargs='*', default=[128, 256, 512],
                        help='--approx-check simhash bit counts (default 128 256 512; none: skip simhash)')
    parser.add_argument('--approx-dims', type=int, nargs='*', default=[32, 48, 64, 128],
                        help='--approx-check PCA dimensions (default 32 48 64 128)')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--simulate', action='store_true',
                        help='explicitly use ideal grouping instead of checking/downloading real MP-SPDZ')
    parser.add_argument('--out', type=Path, help='write setup.json and setup.csv in this directory')
    args = parser.parse_args(argv)
    if args.class_subsets in ('none', ''):
        args.class_subsets = None                                      # the original Dirichlet partition
    args.methods = ['fuzzy' if m == 'similar' else m for m in args.methods]
    if args.group_version is None:
        args.group_version = 2 if args.pair_protocol in ('hegc', 'simhash') else 1
    if args.pair_protocol in ('hegc', 'simhash') and args.group_version != 2:
        parser.error('--pair-protocol hegc/simhash requires --group-version 2')
    if args.simhash_bits < 64 or args.simhash_bits % 64 or (args.pca_dim is not None and args.pca_dim < 1):
        parser.error('--simhash-bits must be a positive multiple of 64 and --pca-dim positive')
    if args.simulate and args.mpc_mode != 'global':
        parser.error('--simulate supports only --mpc-mode global; pairwise requires real MPC')
    if (args.pair_concurrency < 1 or args.pair_workers < 0 or args.group_block_rows < 1
            or not math.isfinite(args.mpc_timeout) or args.mpc_timeout <= 0):
        parser.error('--pair-concurrency, --group-block-rows and --mpc-timeout must be positive, --pair-workers >= 0')
    if args.data == 'real' and args.labels is not None:
        parser.error('--labels is synthetic only; real label spaces come from the original data partitions')
    if args.data == 'real' and min(args.clients) < len(args.datasets.split(',')) + args.num_train_cifar10stl10:
        parser.error('real data needs >= one client per dataset + the special clients')
    if min(args.clients) < 2 or min(args.labels or [3]) < 1 or args.repeats < 1 or args.workers < 1 or args.samples_per_label < 1:
        parser.error('need >=2 clients and positive labels, repeats, and workers')
    if any(not lang.strip() for lang in args.fuzzy_langs.split(',')):
        parser.error('--fuzzy-langs must contain nonempty keyword writers')
    if args.bucket_bits is None:
        args.bucket_bits = 20 if args.data == 'real' else 16
    if not 8 <= args.bucket_bits <= 24:
        parser.error('--bucket-bits must be between 8 and 24')
    if args.mpc_model and (args.pair_protocol != 'hegc' or args.group_protocol != 'atlas' or args.mpc_mode != 'pairwise'):
        parser.error('--mpc-model models --pair-protocol hegc --group-protocol atlas --mpc-mode pairwise')
    secure = any(m != 'plain' for m in args.methods) and not args.approx_check and not args.mpc_model
    if not args.simulate and min(args.clients) < 3 and secure:
        parser.error('MP-SPDZ needs at least 3 clients')
    if args.simulate:
        os.environ.pop('MPSPDZ', None)
    elif secure:
        try:
            root = ensure_mpspdz()
            if args.mpc_mode in {'pairwise', 'both'}:
                gc = f'{args.gc_protocol}-party.x'
                pair_binaries = {'hegc': [gc, 'hemi-party.x'], 'simhash': [gc]}.get(
                    args.pair_protocol, [f'{args.pair_protocol}-party.x'])
                ensure_pairwise_backend(root, (*pair_binaries, f'{args.group_protocol}-party.x'))
            ensure_certificates(root, max(args.clients))
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            parser.error(str(error))
        print(f'Using real MP-SPDZ: {root}', flush=True)

    notes = [
        'Setup excludes data loading/partitioning/sampling, imports, model initialization, training, and evaluation; data preparation times are reported separately.',
        'Real data uses original declared label spaces, including labels without local samples and their original image fallback; labels_per_client is the maximum/padding size.',
        'MP-SPDZ installation/download/native build and certificate setup are excluded from setup timings.',
        'Wall time is a single-host simulation, not network communication latency. No isolated communication timer exists.',
        'Global-mode byte totals use existing protocol accounting plus MPC estimates, even when MP-SPDZ is enabled. Pairwise setup byte estimates are unknown (null), not zero.',
        'mpc_measured reports actual MP-SPDZ traffic/time separately; pairwise global_MB sums matching and grouping traffic. Setup wall time includes compilation and full cryptographic preprocessing.',
        'Pairwise matching sessions follow public batches constrained by pair-concurrency and pair-workers; global private grouping follows matching.',
        'Pairwise mode is hybrid two-party matching plus global honest-majority Shamir grouping; grouping retains the original outputs and convergence leakage.',
        'Private inputs and party processes are colocated on this benchmark host; this is not an isolated multi-host deployment.',
        'Hybrid grouping blocks working buffers, but retains a dense (clients * max_labels)^2 adjacency matrix per party; 50-client full-label runs remain resource intensive.',
        'Compiled circuits are cached by source/compiler/parameters; compile_cache_hits distinguishes cold compilation from cache reuse.',
        'Hybrid matching uses edaBits; grouping defaults to arithmetic-only comparisons because mixed Shamir/CCD preprocessing was expensive in the local scaling smoke test. --group-edabit enables comparison.',
        'Plain byte counts are zero in existing accounting; plaintext label transport is not instrumented.',
        'Public image anchors are cold per trial but shared in-process; text-encoder cache is not cleared.',
        'Compression affects training uploads, so compressed/uncompressed setup is the same and is not duplicated.',
    ]
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  notes=notes, inputs=[], results=[])
    print('Setup only (no training); byte estimates and measured MPC traffic are reported separately.')
    print(f"{'method':<8} {'mode':<8} {'n':>3} {'labels':>6} {'U':>5} {'setup s':>10} {'up B/client':>14} {'down B/client':>14}  backend")
    for n in args.clients:
        for m in (args.labels or [3]) if args.data == 'synthetic' else [None]:
            if args.data == 'real':
                print(f'Preparing real data for {n} clients...', flush=True)
                labels, samples, keywords, data_info = real_inputs(n, args)
            else:
                labels, samples = fixture(n, m, args.seed)
                keywords, data_info = None, dict(source='synthetic', clients=n)
            report['inputs'].append(data_info)
            m = max(map(len, labels))
            if args.pad_to and args.pad_to < m:
                raise SystemExit(f'--pad-to {args.pad_to} < {m} labels of the largest client')
            if args.approx_check:
                for line in approx_check(labels, samples, keywords, args.approx_bits, args.approx_dims):
                    line.update(clients=n)
                    report.setdefault('approx_check', []).append(line)
                    print(json.dumps(line), flush=True)
                if args.out:
                    args.out.mkdir(parents=True, exist_ok=True)
                    (args.out / 'approx_check.json').write_text(json.dumps(report, indent=2) + '\n')
                continue
            for repeat in range(args.repeats):
                for method in args.methods:
                    modes = (['global'] if method == 'plain' else
                             ['global', 'pairwise'] if args.mpc_mode == 'both' else [args.mpc_mode])
                    per_client = args.pair_workers == 0                 # one session per client at a time
                    workers = max(1, n // 2) if per_client else args.pair_workers
                    concurrency = 1 if per_client else args.pair_concurrency
                    for mode in modes:
                        print(f'Starting {method}/{mode}: {n} clients, max {m} labels/client, trial {repeat + 1}/{args.repeats}', flush=True)
                        try:
                            row = benchmark(method, labels, samples, args.bucket_bits, args.workers, keywords=keywords,
                                            mpc_mode=mode, pair_concurrency=concurrency,
                                            pair_workers=workers, mpc_timeout=args.mpc_timeout,
                                            group_prefix=args.group_prefix, group_block_rows=args.group_block_rows,
                                            group_edabit=args.group_edabit, group_version=args.group_version,
                                            group_protocol=args.group_protocol, pair_protocol=args.pair_protocol,
                                            pca_dim=args.pca_dim, simhash_bits=args.simhash_bits,
                                            simhash_u0=args.simhash_u0, gc_protocol=args.gc_protocol,
                                            pad_max=args.pad_max, mpc_model=args.mpc_model, pad_to=args.pad_to)
                        except (RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
                            # Keep completed trials and public diagnostics even on the first failure.
                            # Never serialize input rows, private outputs, or raw process logs.
                            report.setdefault('failures', []).append(dict(clients=n, labels_per_client=m,
                                method=method, mpc_mode=mode, repeat=repeat + 1,
                                error=str(error) if isinstance(error, MPCSessionError) else type(error).__name__))
                            if args.out:
                                args.out.mkdir(parents=True, exist_ok=True)
                                temporary = args.out / 'setup.json.tmp'
                                temporary.write_text(json.dumps(report, indent=2) + '\n')
                                temporary.replace(args.out / 'setup.json')
                            raise
                        row.update(data_source=args.data, data_load_partition_seconds=data_info.get('data_load_partition_seconds'),
                                   sampling_seconds=data_info.get('sampling_seconds'))
                        row['repeat'] = repeat + 1
                        row.update(setup_comm(row, args.net_mbps, args.net_rtt_ms, workers, per_client))
                        if row['backend'] == 'mpc-model':             # plaintext part measured + modelled MPC
                            row['setup_compute_seconds_model'] = row['setup_wall_seconds'] + row['mpc_wall_seconds']
                            row['setup_total_seconds_model'] = (row['setup_compute_seconds_model']
                                                                + row['comm_seconds_model_max'])
                        report['results'].append(row)
                        if args.out:
                            save_report(args.out, report)
                        up = row['estimated_upload_bytes_per_client']
                        down = row['estimated_download_bytes_per_client']
                        up_text = 'unknown' if up is None else f'{up:.0f}'
                        down_text = 'unknown' if down is None else f'{down:.0f}'
                        if row['backend'] == 'mpc-model':
                            print(f"{method:<8} {mode:<8} {n:>3} {m:>6} {row['union_size']:>5}  MODEL: MPC "
                                  f"{row['mpc_global_MB']:.0f} MB total, {row['comm_MB_per_client_mean']:.0f} MB/client, "
                                  f"compute {row['setup_compute_seconds_model']:.1f} s + comm {row['comm_seconds_model_max']:.1f} s "
                                  f"= {row['setup_total_seconds_model']:.1f} s, MCC {row.get('pair_mcc', float('nan')):.3f}", flush=True)
                            continue
                        print(f"{method:<8} {mode:<8} {n:>3} {m:>6} {row['union_size']:>5} {row['setup_wall_seconds']:>10.4f} "
                              f"{up_text:>14} {down_text:>14}  {row['backend']}  comm {row['comm_MB_per_client_mean']:.1f} MB/client, "
                              f"model {row['comm_seconds_model_max']:.1f} s"
                              + (f", MCC {row['pair_mcc']:.3f}" if 'pair_mcc' in row else ''), flush=True)
    if args.out:
        print(f'Results: {args.out / "setup.json"} and {args.out / "setup.csv"}')
    for note in notes:
        print('Note: ' + note)


if __name__ == '__main__':
    main()
