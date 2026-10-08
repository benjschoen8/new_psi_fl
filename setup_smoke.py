"""Setup-only no-cluster benchmark; no models or training are constructed.

    python -m setup_smoke
    python -m setup_smoke --clients 3 5 --repeats 3 --out setup_results
    python -m setup_smoke --data synthetic --clients 3 --labels 3
    MPSPDZ=/path/to/mp-spdz python -m setup_smoke --methods plain exact fuzzy

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
    try:
        if not _valid_mpspdz(root):
            if platform.system() == 'Linux' and shutil.which('apt-get') and shutil.which('dpkg-query'):
                packages = ('automake build-essential clang cmake git libboost-dev libboost-filesystem-dev '
                            'libboost-iostreams-dev libboost-thread-dev libgmp-dev libntl-dev libsodium-dev '
                            'libssl-dev libtool python3 ca-certificates openssl').split()
                missing = []
                for package in packages:
                    status = subprocess.run(['dpkg-query', '-W', '-f=${Status}', package],
                                            capture_output=True, text=True)
                    if status.returncode or status.stdout.strip() != 'install ok installed':
                        missing.append(package)
                if missing:
                    sudo = [] if os.geteuid() == 0 else ['sudo']
                    print('Installing build dependencies; sudo may ask for your password.', file=sys.stderr)
                    subprocess.run(sudo + ['apt-get', 'update'], check=True, stdout=sys.stderr)
                    subprocess.run(sudo + ['apt-get', 'install', '-y', *missing], check=True, stdout=sys.stderr)
            for tool in ('git', 'make', 'clang++', 'cmake', 'python3', 'openssl'):
                if shutil.which(tool) is None:
                    raise RuntimeError(f'Building MP-SPDZ needs {tool}. Install its build dependencies first.')
            ref = os.environ.get('MPSPDZ_REF', 'v0.4.2')
            if not (root / '.git').is_dir():
                root.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['git', 'clone', '--depth', '1', '--branch', ref, MPSPDZ_REPO, str(root)],
                               check=True, stdout=sys.stderr)
            jobs = os.environ.get('MPSPDZ_JOBS', '4')
            print('Compiling native MP-SPDZ (first build can take 10-30+ minutes)...', file=sys.stderr)
            subprocess.run(['make', 'setup'], cwd=root, check=True, stdout=sys.stderr)
            subprocess.run(['make', '-j', jobs, 'shamir-party.x'], cwd=root, check=True, stdout=sys.stderr)
        # Native source builds need the same certificates as binary installations.
        # Retry this stage after an interrupted build/setup, even if the binary exists.
        if not (root / 'Player-Data' / 'P15.pem').is_file():
            subprocess.run(['bash', 'Scripts/setup-ssl.sh', '16'], cwd=root, check=True, stdout=sys.stderr)
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


def ensure_certificates(root, parties):
    """Prepare client TLS credentials before timing; installers initially create only 16."""
    if any(not (root / 'Player-Data' / f'P{i}.{suffix}').is_file()
           for i in range(parties) for suffix in ('pem', 'key')):
        print(f'Preparing MP-SPDZ certificates for {parties} parties...', file=sys.stderr)
        subprocess.run(['bash', 'Scripts/setup-ssl.sh', str(parties)],
                       cwd=root, check=True, stdout=sys.stderr)


def save_report(directory, report):
    """Keep completed trials after a later failure; replace each output atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'setup.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(directory / 'setup.json')
    fields = [k for k in report['results'][0]
              if k not in {'protocol_stats', 'mpc_measured', 'secagg_accounted_bytes'}]
    temporary = directory / 'setup.csv.tmp'
    with temporary.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['results'])
    temporary.replace(directory / 'setup.csv')


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


def real_inputs(n, args):
    """Reuse the training CLI's data components without build_clients/model construction."""
    from setup import parser, seed_all, label_names, label_samples
    from fl_datasets import load_partitioned_datasets
    from omegaconf import OmegaConf
    from rt_descriptions import keyword

    start = time.perf_counter()
    data_args = parser().parse_args([])
    for key in vars(data_args):
        if key.startswith('num_train_'):
            setattr(data_args, key, 0)
    counts = dict(zip(('MNIST', 'EMNIST', 'CIFAR10'), (n // 3 + (i < n % 3) for i in range(3))))
    for name, count in counts.items():
        setattr(data_args, 'num_train_' + name.lower(), count)
    for key in ('seed', 'data_root', 'class_subsets', 'class_share', 'noniid_partition'):
        setattr(data_args, key, getattr(args, key))
    seed_all(args.seed)
    config = OmegaConf.to_container(OmegaConf.load(args.exp_conf), resolve=True)
    if config.get('channels', 3) != 3 or config.get('img_size', 32) != 32:
        raise ValueError('Existing dataset transforms require channels=3 and img_size=32')
    loaders, _, _ = load_partitioned_datasets(data_args, str(args.data_root), **config)
    loaded = time.perf_counter()
    labels, samples, keywords, dataset_of = [], [], [], []
    langs = args.fuzzy_langs.split(',')
    for dataset, entries in loaders.items():
        for entry in entries:
            names = list(label_names(entry['train'].dataset, dataset))
            images = label_samples(entry['train'], names, args.samples_per_label)
            if not images:
                raise ValueError(f'{dataset} client {len(labels)} has no training images for setup')
            lang = langs[len(labels) % len(langs)]
            keywords.append({x: keyword(dataset, x, lang) for x in names})
            labels.append(names)
            samples.append(images)
            dataset_of.append(dataset)
    if len(labels) != n:
        raise ValueError(f'Expected {n} clients, loader returned {len(labels)}')
    return labels, samples, keywords, dict(
        source='real', clients=n, dataset_clients=counts, dataset_of_client=dataset_of,
        client_label_counts=list(map(len, labels)), client_sampled_label_counts=list(map(len, samples)),
        data_load_partition_seconds=loaded - start, sampling_seconds=time.perf_counter() - loaded,
        samples_per_label=args.samples_per_label, partition_config=config)


def benchmark(method, labels, samples, bucket_bits=16, workers=1, keywords=None):
    # Make every trial include cold public image-anchor construction. The single
    # process then shares that cache, as in the current simulation, not n hosts.
    anchor_matrix.cache_clear()
    start = time.perf_counter()
    sets = [client_sets(s, names) for s, names in zip(samples, labels)]
    image_seconds = time.perf_counter() - start
    union_start = time.perf_counter()
    _, _, _, U, stats = circuit_union_with_keys(
        labels, keywords if method == 'fuzzy' and keywords is not None else [{x: x for x in own} for own in labels], sets,
        fuzzy=method == 'fuzzy', secure=method != 'plain',
        bucket_bits=bucket_bits, workers=workers)
    elapsed = time.perf_counter() - start
    union_seconds = time.perf_counter() - union_start
    mpc = stats.get('mpc', {})
    measured = mpc.get('measured')
    return dict(
        method=method, clients=len(labels), labels_per_client=max(map(len, labels)),
        labels_per_client_min=min(map(len, labels)), labels_per_client_mean=float(np.mean(list(map(len, labels)))),
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
    parser.add_argument('--clients', type=int, nargs='+', default=[3, 5, 10, 30, 50],
                        help='client counts to benchmark sequentially (default: 3 5 10 30 50)')
    parser.add_argument('--data', choices=['real', 'synthetic'], default='real')
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--exp-conf', type=Path, default=Path('config.yaml'))
    parser.add_argument('--samples-per-label', type=int, default=16)
    parser.add_argument('--class-subsets', help='same as training CLI: LO,HI or even; default uses original Non-IID partition')
    parser.add_argument('--class-share', choices=['split', 'full'], default='split')
    parser.add_argument('--noniid-partition', default='dirichlet',
                        choices=['dirichlet', 'noniid_label', 'quantity_skew', 'quantity_skew_equalSize'])
    parser.add_argument('--fuzzy-langs', default='en0,en1', help='same client keyword writers as training CLI')
    parser.add_argument('--labels', type=int, nargs='+', help='synthetic only: labels per client (default 3)')
    parser.add_argument('--methods', nargs='+', choices=['plain', 'exact', 'fuzzy'], default=['plain', 'exact'])
    parser.add_argument('--bucket-bits', type=int,
                        help='default 20 for real data (production size), 16 for synthetic')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--simulate', action='store_true',
                        help='explicitly use ideal grouping instead of checking/downloading real MP-SPDZ')
    parser.add_argument('--out', type=Path, help='write setup.json and setup.csv in this directory')
    args = parser.parse_args(argv)
    if args.data == 'real' and args.labels is not None:
        parser.error('--labels is synthetic only; real label spaces come from the original data partitions')
    if args.data == 'real' and min(args.clients) < 3:
        parser.error('real mixed data needs >=3 clients (at least one per dataset)')
    if min(args.clients) < 2 or min(args.labels or [3]) < 1 or args.repeats < 1 or args.workers < 1 or args.samples_per_label < 1:
        parser.error('need >=2 clients and positive labels, repeats, and workers')
    if any(not lang.strip() for lang in args.fuzzy_langs.split(',')):
        parser.error('--fuzzy-langs must contain nonempty keyword writers')
    if args.bucket_bits is None:
        args.bucket_bits = 20 if args.data == 'real' else 16
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
            ensure_certificates(root, max(args.clients))
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            parser.error(str(error))
        print(f'Using real MP-SPDZ: {root}', flush=True)

    notes = [
        'Setup excludes data loading/partitioning/sampling, imports, model initialization, training, and evaluation; data preparation times are reported separately.',
        'Real data uses original declared label spaces, including labels without local samples and their original image fallback; labels_per_client is the maximum/padding size.',
        'MP-SPDZ installation/download/native build and certificate setup are excluded from setup timings.',
        'Wall time is a single-host simulation, not network communication latency. No isolated communication timer exists.',
        'Byte totals use the existing protocol accounting plus MPC estimates, even when MP-SPDZ is enabled.',
        'mpc_measured reports actual MP-SPDZ grouping traffic/time separately; setup wall time includes its compilation.',
        'Plain byte counts are zero in existing accounting; plaintext label transport is not instrumented.',
        'Public image anchors are cold per trial but shared in-process; text-encoder cache is not cleared.',
        'Compression affects training uploads, so compressed/uncompressed setup is the same and is not duplicated.',
    ]
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  notes=notes, inputs=[], results=[])
    print('Setup only (no training); bytes are protocol estimates, not measured network traffic.')
    print(f"{'method':<8} {'n':>3} {'labels':>6} {'U':>5} {'setup s':>10} {'up B/client':>14} {'down B/client':>14}  backend")
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
            for repeat in range(args.repeats):
                for method in args.methods:
                    print(f'Starting {method}: {n} clients, max {m} labels/client, trial {repeat + 1}/{args.repeats}', flush=True)
                    row = benchmark(method, labels, samples, args.bucket_bits, args.workers, keywords=keywords)
                    row.update(data_source=args.data, data_load_partition_seconds=data_info.get('data_load_partition_seconds'),
                               sampling_seconds=data_info.get('sampling_seconds'))
                    row['repeat'] = repeat + 1
                    report['results'].append(row)
                    if args.out:
                        save_report(args.out, report)
                    print(f"{method:<8} {n:>3} {m:>6} {row['union_size']:>5} {row['setup_wall_seconds']:>10.4f} "
                          f"{row['estimated_upload_bytes_per_client']:>14.0f} "
                          f"{row['estimated_download_bytes_per_client']:>14.0f}  {row['backend']}", flush=True)
    if args.out:
        print(f'Results: {args.out / "setup.json"} and {args.out / "setup.csv"}')
    for note in notes:
        print('Note: ' + note)


if __name__ == '__main__':
    main()
