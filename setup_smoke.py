"""Setup-only, synthetic no-cluster benchmark; no models or training are constructed.

    python -m setup_smoke
    python -m setup_smoke --clients 3 5 --labels 3 10 --repeats 3 --out setup_results
    MPSPDZ=/path/to/mp-spdz python -m setup_smoke --methods plain exact fuzzy

Setup means image signatures/anchors, label grouping, bucket union, and public-key
distribution. Model initialization, data loading, training, and evaluation are excluded.
Fuzzy uses the existing text encoder (cache or optional sentence-transformers).

On x86-64 Linux a prebuilt MP-SPDZ binary is downloaded. On other platforms (ARM,
macOS, ...) MP-SPDZ is cloned and compiled from source into ~/.cache/mp-spdz
(override with MPSPDZ_BUILD_DIR; pin a tag/branch with MPSPDZ_REF).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from label_union.circuit_union import circuit_union_with_keys
from label_union.domain import anchor_matrix, client_sets

MPSPDZ_REPO = 'https://github.com/data61/MP-SPDZ.git'


def _valid_mpspdz(root):
    return ((root / 'compile.py').is_file() and (root / 'shamir-party.x').is_file()
            and os.access(root / 'shamir-party.x', os.X_OK))


def build_mpspdz_from_source():
    """Clone and compile MP-SPDZ locally (needed on ARM, where no binary release exists)."""
    root = Path(os.environ.get('MPSPDZ_BUILD_DIR',
                               Path.home() / '.cache' / 'mp-spdz')).expanduser().resolve()
    if _valid_mpspdz(root):
        return root
    for tool in ('git', 'make', 'g++'):
        if shutil.which(tool) is None and not (tool == 'g++' and shutil.which('clang++')):
            raise RuntimeError(f'Building MP-SPDZ needs {tool}. Install the build dependencies first '
                               '(see the README of MP-SPDZ).')
    ref = os.environ.get('MPSPDZ_REF')  # optionally pin a tag/branch, e.g. v0.4.0
    try:
        if not (root / '.git').is_dir():
            root.parent.mkdir(parents=True, exist_ok=True)
            cmd = ['git', 'clone', '--depth', '1']
            if ref:
                cmd += ['--branch', ref]
            subprocess.run(cmd + [MPSPDZ_REPO, str(root)], check=True, stdout=sys.stderr)
        jobs = str(os.cpu_count() or 2)
        print('Compiling MP-SPDZ from source (this can take 10-30+ minutes)...', file=sys.stderr)
        subprocess.run(['make', 'setup'], cwd=root, check=True, stdout=sys.stderr)
        subprocess.run(['make', '-j', jobs, 'shamir-party.x'], cwd=root, check=True, stdout=sys.stderr)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Building MP-SPDZ from source failed (see output above). Install its '
                           'dependencies, or set MPSPDZ to a working installation, or use --simulate.') from error
    return root


def ensure_mpspdz():
    """Reuse a valid configured install; otherwise use the Linux binary (x86-64) or build from source (ARM etc.)."""
    configured = os.environ.get('MPSPDZ')
    if configured:
        root = Path(configured).expanduser().resolve()
        if _valid_mpspdz(root):
            os.environ['MPSPDZ'] = str(root)
            return root
        print(f'MPSPDZ={configured} is incomplete; trying to install.', file=sys.stderr)

    machine = platform.machine().lower()
    if platform.system() == 'Linux' and machine in {'x86_64', 'amd64'}:
        installer = Path(__file__).resolve().with_name('get_mpspdz.sh')
        try:
            result = subprocess.run(['bash', str(installer)], stdout=subprocess.PIPE, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError('MP-SPDZ binary installation failed (see installer output above). '
                               'Set MPSPDZ to a working installation or use --simulate.') from error
        root = Path(result.stdout.strip()).expanduser().resolve()
    else:
        print(f'No prebuilt MP-SPDZ for {platform.system()}/{machine}; building from source.', file=sys.stderr)
        root = build_mpspdz_from_source()

    if not _valid_mpspdz(root):
        raise RuntimeError('MP-SPDZ install did not produce a directory containing compile.py '
                           'and executable shamir-party.x')
    os.environ['MPSPDZ'] = str(root)
    return root


def fixture(n, m, seed):
    """Equal-sized clients with overlapping names and identical images for shared labels."""
    rng = np.random.default_rng(seed)
    vocabulary = ['cat', 'dog', 'car', 'truck', 'bird', 'horse', 'ship', 'plane', 'frog', 'deer']
    count = m + max(1, m // 2)
    vocabulary += [f'object {i}' for i in range(len(vocabulary), count)]
    pool = vocabulary[:count]
    images = {x: rng.uniform(-1, 1, (16, 3, 32, 32)) for x in pool}
    labels = [[pool[(c + j) % count] for j in range(m)] for c in range(n)]
    samples = [{x: images[x] for x in own} for own in labels]
    return labels, samples


def benchmark(method, labels, samples, bucket_bits=16, workers=1):
    # Make every trial include cold public image-anchor construction. The single
    # process then shares that cache, as in the current simulation, not n hosts.
    anchor_matrix.cache_clear()
    start = time.perf_counter()
    sets = [client_sets(s, names) for s, names in zip(samples, labels)]
    image_seconds = time.perf_counter() - start
    union_start = time.perf_counter()
    _, _, _, U, stats = circuit_union_with_keys(
        labels, [{x: x for x in own} for own in labels], sets,
        fuzzy=method == 'fuzzy', secure=method != 'plain',
        bucket_bits=bucket_bits, workers=workers)
    elapsed = time.perf_counter() - start
    union_seconds = time.perf_counter() - union_start
    mpc = stats.get('mpc', {})
    measured = mpc.get('measured')
    return dict(
        method=method, clients=len(labels), labels_per_client=max(map(len, labels)),
        bucket_bits=bucket_bits, union_size=U,
        backend=('plain' if method == 'plain' else
                 'mp-spdz' if measured else 'ideal-functionality'),
        setup_wall_seconds=elapsed, image_setup_seconds=image_seconds,
        union_wall_seconds=union_seconds,
        mpc_compile_seconds=(measured or {}).get('compile_seconds'),
        mpc_wall_seconds=(measured or {}).get('wall_seconds'),
        mpc_global_MB=(measured or {}).get('global_MB'),
        estimated_upload_bytes_per_client=stats['setup_upload_bytes_per_client'],
        estimated_download_bytes_per_client=stats['setup_download_bytes_per_client'],
        estimated_upload_bytes_total=len(labels) * stats['setup_upload_bytes_per_client'],
        estimated_download_bytes_total=len(labels) * stats['setup_download_bytes_per_client'],
        secagg_accounted_bytes=stats.get('secagg'), mpc_measured=measured,
        protocol_stats=stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clients', type=int, nargs='+', default=[3])
    parser.add_argument('--labels', type=int, nargs='+', default=[3], help='labels per client')
    parser.add_argument('--methods', nargs='+', choices=['plain', 'exact', 'fuzzy'], default=['plain', 'exact'])
    parser.add_argument('--bucket-bits', type=int, default=16,
                        help='16 for smoke; use 20 for the production bucket-vector size')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--simulate', action='store_true',
                        help='explicitly use ideal grouping instead of checking/downloading real MP-SPDZ')
    parser.add_argument('--out', type=Path, help='write setup.json and setup.csv in this directory')
    args = parser.parse_args(argv)
    if min(args.clients) < 2 or min(args.labels) < 1 or args.repeats < 1 or args.workers < 1:
        parser.error('need >=2 clients and positive labels, repeats, and workers')
    if not 8 <= args.bucket_bits <= 24:
        parser.error('--bucket-bits must be between 8 and 24')
    secure = any(m != 'plain' for m in args.methods)
    if not args.simulate and min(args.clients) < 3 and secure:
        parser.error('MP-SPDZ needs at least 3 clients')
    if args.simulate:
        os.environ.pop('MPSPDZ', None)
    elif secure:
        try:
            root = ensure_mpspdz()
        except RuntimeError as error:
            parser.error(str(error))
        print(f'Using real MP-SPDZ: {root}', flush=True)

    notes = [
        'Synthetic setup only; excludes fixture generation, imports, model initialization, training, and evaluation.',
        'MP-SPDZ installation/download/compile time is excluded from setup timings.',
        'Wall time is a single-host simulation, not network communication latency. No isolated communication timer exists.',
        'Byte totals use the existing protocol accounting plus MPC estimates, even when MP-SPDZ is enabled.',
        'mpc_measured reports actual MP-SPDZ grouping traffic/time separately; setup wall time includes its compilation.',
        'Plain byte counts are zero in existing accounting; plaintext label transport is not instrumented.',
        'Public image anchors are cold per trial but shared in-process; text-encoder cache is not cleared.',
        'Compression affects training uploads, so compressed/uncompressed setup is the same and is not duplicated.',
    ]
    report = dict(config={**vars(args), 'out': str(args.out) if args.out else None}, notes=notes, results=[])
    print('Setup only (no training); bytes are protocol estimates, not measured network traffic.')
    print(f"{'method':<8} {'n':>3} {'labels':>6} {'U':>5} {'setup s':>10} {'up B/client':>14} {'down B/client':>14}  backend")
    for n in args.clients:
        for m in args.labels:
            labels, samples = fixture(n, m, args.seed)
            for repeat in range(args.repeats):
                for method in args.methods:
                    row = benchmark(method, labels, samples, args.bucket_bits, args.workers)
                    row['repeat'] = repeat + 1
                    report['results'].append(row)
                    print(f"{method:<8} {n:>3} {m:>6} {row['union_size']:>5} {row['setup_wall_seconds']:>10.4f} "
                          f"{row['estimated_upload_bytes_per_client']:>14.0f} "
                          f"{row['estimated_download_bytes_per_client']:>14.0f}  {row['backend']}", flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'setup.json').write_text(json.dumps(report, indent=2) + '\n')
        fields = [k for k in report['results'][0] if k not in {'protocol_stats', 'mpc_measured', 'secagg_accounted_bytes'}]
        with (args.out / 'setup.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['results'])
        print(f'Results: {args.out / "setup.json"} and {args.out / "setup.csv"}')
    for note in notes:
        print('Note: ' + note)


if __name__ == '__main__':
    main()