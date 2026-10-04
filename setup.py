"""CLI composition, dataset metadata and artifact persistence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from datetime import datetime, timezone


def label_names(dataset, dataset_name):
    """Resolve semantic names in the *actual* local index order of a dataset."""
    if hasattr(dataset, 'mapping_dict'):
        original = label_names(dataset.dataset, dataset_name)
        permutation = dataset.mapping_dict
        if set(permutation) != set(range(len(original))) or set(permutation.values()) != set(permutation):
            raise ValueError('Expected a bijective local label permutation')
        names = [None] * len(original)
        for old, new in permutation.items():
            names[new] = original[old]
        return tuple(names)
    if hasattr(dataset, 'dataset'):
        return label_names(dataset.dataset, dataset_name)
    if dataset_name in ('MNIST', 'USPS', 'SVHN'):
        return tuple(str(i) for i in range(10))
    if not hasattr(dataset, 'classes'):
        raise ValueError(f'No explicit class metadata for {dataset_name}')
    return tuple(str(name) for name in dataset.classes)


def resolve_device(name='auto'):
    """'auto' picks CUDA if present, then Apple MPS, then CPU; anything else is used as given."""
    import torch
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def sync(device):
    """Wait for queued GPU work so wall-clock timings are real."""
    import torch
    kind = str(device).split(':')[0]
    if kind == 'cuda':
        torch.cuda.synchronize(device)
    elif kind == 'mps':
        torch.mps.synchronize()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                      # also seeds CUDA and MPS generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunLogger:
    def __init__(self, path):
        self.path = path

    def log(self, text):
        print(text)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(str(text) + '\n')


def build_clients(args, config):
    from fl_datasets import load_partitioned_datasets
    from nets import get_heterogeneous_model, DCGANGenerator, DCGANDiscriminator, ResNet, BasicBlock
    from client import Client
    loaders, _, _ = load_partitioned_datasets(args, str(args.data_root), **config)
    if config.get('channels', 3) != 3 or config.get('img_size', 32) != 32:
        raise ValueError('Existing dataset transforms require channels=3 and img_size=32')
    def generator_factory(count):
        return DCGANGenerator(count, noise_dim=config.get('gen_noise_dim', 128), img_size=32, channels=3)
    def classifier_factory(count):
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=count, global_dim=256)
    clients, spaces, tests, metadata = [], {}, [], {}
    for name, entries in loaders.items():
        for index, entry in enumerate(entries[:len(entries) - args.num_new_clients]):
            cid = len(clients)
            names = label_names(entry['train'].dataset, name)
            if names != label_names(entry['test'].dataset, name):
                raise ValueError(f'{name} train/test label orders differ')
            arch = index % 10 if config.get('heterogeneous', False) else 1
            model = get_heterogeneous_model(arch, in_channels=3, num_classes=len(names),
                                           img_size=32, global_dim=config.get('global_feature_dim', 256))
            clients.append(Client(cid, model, generator_factory(len(names)),
                                  DCGANDiscriminator(len(names), img_size=32, channels=3),
                                  entry['train'], len(names), config, args.device))
            spaces[cid] = names
            tests.append((name, cid, entry['test']))
            metadata[cid] = dict(dataset=name, architecture=arch, labels=names,
                                 train_indices=[int(i) for i in entry['train'].dataset.indices],
                                 test_indices=[int(i) for i in entry['test'].dataset.indices])
    return clients, spaces, tests, metadata, generator_factory, classifier_factory


def parser():
    p = argparse.ArgumentParser(description='image_bi: PACFL → GeFL with independent participants')
    p.add_argument('--exp-conf', '--exp_conf', type=Path, default=Path('config.yaml'))
    p.add_argument('--data-root', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='auto', help='auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu')
    p.add_argument('--seed', type=int, default=15698)
    p.add_argument('--mapping', choices=('image_bi', 'by_class', 'psi_trivial_circuit', 'fuzzy_psi_circuit',
                                            'psi_trivial_circuit_with_key', 'fuzzy_psi_circuit_with_key'), default='image_bi')
    p.add_argument('--rounds', type=int)
    p.add_argument('--mapping-round', type=int)
    p.add_argument('--clustering', choices=('pacfl', 'none'), default='pacfl',
                   help='none: every client uploads its own generator, no averaging across clients')
    p.add_argument('--pacfl-basis-budget', type=int, default=20)
    p.add_argument('--pacfl-cluster-alpha', type=float, default=20)
    p.add_argument('--pretrained', type=Path, help='Warm-start clients from a revised_protocol checkpoint')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic CPU-capable experiment; no downloads')
    p.add_argument('--num-new-clients', '--num_new_clients', type=int, default=0)
    for name in ('mnist', 'emnist', 'fashionmnist', 'cifar10', 'cifar100', 'usps', 'svhn', 'stl10'):
        p.add_argument(f'--num-train-{name}', f'--num_train_{name}', type=int,
                       default=10 if name in ('mnist', 'emnist', 'cifar10') else 0)
    p.add_argument('--class-subsets', metavar='LO,HI|even',
                   help='every client of a dataset holds LO..HI of its classes (even coverage, each class '
                        'split evenly among its holders) and knows only those labels; e.g. 8,20. '
                        '"even": every class held by 2 clients, classes and images spread evenly')
    p.add_argument('--class-share', choices=('split', 'full'), default='split',
                   help='--class-subsets: split = a class\'s images split evenly among its holders; '
                        'full = every holder gets all images of its classes')
    p.add_argument('--noniid-partition', '--noniid_partition', default='dirichlet',
                   choices=('dirichlet', 'noniid_label', 'quantity_skew', 'quantity_skew_equalSize'))
    return p


def run_cli(run):
    args = parser().parse_args()
    args.device = resolve_device(args.device)
    import torch
    from omegaconf import OmegaConf
    from aggregation import GeFLAggregation
    from server import Server
    from mapping import (ImageBiMapping, PSITrivialCircuit, FuzzyPSICircuit,
                         PSITrivialCircuitWithKey, FuzzyPSICircuitWithKey)
    from training import GlobalClassifierTrainer
    from main import RunConfig
    seed_all(args.seed)
    config = OmegaConf.to_container(OmegaConf.load(args.exp_conf), resolve=True)
    if args.smoke:
        config.update(global_rounds=2, start_mapping_epoch=1, local_epochs=1,
                      gen_local_epochs=1, global_model_epochs=1, batch_size=4,
                      global_samples_per_class=4, gen_noise_dim=4)
    run_config = RunConfig(rounds=args.rounds if args.rounds is not None else config.get('global_rounds', 45),
                           mapping_round=args.mapping_round if args.mapping_round is not None else config.get('start_mapping_epoch', 25),
                           sample_fraction=config.get('sample_frac', 1),
                           basis_budget=args.pacfl_basis_budget, seed=args.seed,
                           device=args.device, pretrained=args.pretrained is not None)
    output = args.output or Path('runs') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + args.mapping)
    output.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(output / 'run.log')
    if args.smoke:
        from smoke import build_smoke_clients
        built = build_smoke_clients(config, args.device)
    else:
        built = build_clients(args, config)
    clients, spaces, tests, metadata, generator_factory, classifier_factory = built
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        if checkpoint.get('format') != 'revised_protocol.v1':
            raise ValueError('Expected a v1 checkpoint')
        if checkpoint['metadata'] != metadata:
            raise ValueError('Checkpoint client datasets, label orders, architectures or splits differ')
        for client in clients:
            state = checkpoint['clients'][client.id]
            client.model.load_state_dict(state['classifier'])
            client.generator.load_state_dict(state['generator'])
            client.discriminator.load_state_dict(state['discriminator'])
            client.trained = True
    (output / 'config.json').write_text(json.dumps(dict(
        protocol='image_bi', mapping_strategy=args.mapping, run=asdict(run_config),
        experiment=config, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch_version=str(torch.__version__), client_metadata=metadata), indent=2), encoding='utf-8')
    def record(row):
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        evaluation = row.get('evaluation', {})
        logger.log(f"Round {row['round']}: old_acc={evaluation.get('old_acc', 'pending')}, "
                   f"ground_truth_acc={evaluation.get('ground_truth_acc', 'pending')}")
    server = Server(GeFLAggregation())
    langs = config.get('rt_langs', ['en'])
    if args.mapping == 'by_class':
        mapper = 'by_class'
    elif args.mapping.startswith('psi_trivial_circuit'):
        keyed = args.mapping.endswith('_with_key')
        mapper = (PSITrivialCircuitWithKey if keyed else PSITrivialCircuit)(langs)
    elif args.mapping.startswith('fuzzy_psi_circuit'):
        keyed = args.mapping.endswith('_with_key')
        mapper = (FuzzyPSICircuitWithKey if keyed else FuzzyPSICircuit)(config.get('rt_method', 'filter'), config.get('rt_psi', 'plain'), langs,
                                 config.get('rt_samples', 32), config.get('gen_noise_dim', 128),
                                 device=args.device, seed=args.seed, logger=logger)
    else:
        mapper = ImageBiMapping(
            logger, args.device, config.get('gen_noise_dim', 128), config.get('entropy_ratio', .25),
            config.get('use_new_entropy_method', True))
    if args.clustering == 'pacfl':
        from clustering import PACFL
        clustering = PACFL(args.pacfl_cluster_alpha)
    else:
        clustering = None
    result = run(clients, server, clustering=clustering,
                 mapping_strategy=mapper, global_trainer=GlobalClassifierTrainer(classifier_factory, config, args.device),
                 generator_factory=generator_factory, label_spaces=spaces, test_sets=tests,
                 config=run_config, record=record)
    (output / 'relations.json').write_text(json.dumps({k: result[k] for k in ('groups', 'mapping', 'ground_truth')},
                                                     indent=2), encoding='utf-8')
    from contracts import clone_state
    torch.save(dict(format='revised_protocol.v1', metadata=metadata, protocol='image_bi',
                    groups=result['groups'], mapping=result['mapping'], ground_truth=result['ground_truth'],
                    global_classifier=clone_state(result['model'].state_dict()),
                    clients={c.id: dict(classifier=clone_state(c.model.state_dict()),
                                        generator=clone_state(c.generator.state_dict()),
                                        discriminator=clone_state(c.discriminator.state_dict())) for c in clients},
                    group_gans={g: dict(generator=s.generator, discriminator=s.discriminator)
                                for g, s in server.snapshot().items()}), output / 'checkpoint.pt')
    logger.log(f'Artifacts: {output.resolve()}')
