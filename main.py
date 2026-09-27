"""Composition and orchestration: PACFL → local GeFL → mapping → global evaluation.

Run from the repository root: python -m main --help
"""
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter
import random
import hashlib

from contracts import MappingInputs, payload_bytes
from mapping import ByClassMapping, validate_mapping
from evaluation import evaluate_global, mapping_metrics


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 45
    mapping_round: int = 25
    sample_fraction: float = 1.
    basis_budget: int = 20
    seed: int = 15698
    device: str = 'cpu'
    pretrained: bool = False

    def __post_init__(self):
        if not 1 <= self.mapping_round <= self.rounds:
            raise ValueError('Require 1 <= mapping_round <= rounds')
        if not 0 < self.sample_fraction <= 1:
            raise ValueError('sample_fraction must be in (0, 1]')


def run(clients, server, *, clustering, mapping_strategy, global_trainer,
        generator_factory, label_spaces, test_sets, config=RunConfig(),
        evaluator=evaluate_global, record=None):
    """label_spaces: client ID → ordered canonical semantic names.

    clustering: callable(bases) → {client ID: group}, or None for no clustering
    (one group per client; PACFL bases are not computed or uploaded).

    mapping_strategy is a callable, or the string 'by_class' for the oracle.
    Pass a normal callable for client-to-client methods: the contract contains
    model evidence, not a Server. A server-backed adapter can capture a server
    at construction, without changing the orchestration or other strategies.
    """
    import torch
    ids = [c.id for c in clients]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(label_spaces):
        raise ValueError('Unique clients and complete label metadata required')
    if config.pretrained and any(not c.trained for c in clients):
        raise ValueError('pretrained run requires trained client classifier/GAN states')
    rng = random.Random(config.seed)

    def seed_stage(stage):
        from setup import seed_all
        digest = hashlib.sha256(f'{config.seed}:{stage}'.encode()).digest()
        seed_all(int.from_bytes(digest[:4], 'big'))

    def timed(fn):
        from setup import sync
        sync(config.device)
        start = perf_counter()
        result = fn()
        sync(config.device)
        return result, perf_counter() - start

    if clustering is None:
        # No clustering: each client is its own group, so the server keeps
        # every client's generator unaveraged and trains on all of them.
        bases, basis_seconds, cluster_seconds = {}, 0., 0.
        groups = {c.id: f'Client_{c.id}' for c in clients}
    else:
        seed_stage('basis')
        bases, basis_seconds = timed(lambda: {c.id: c.basis(config.basis_budget) for c in clients})
        seed_stage('clustering')
        groups, cluster_seconds = timed(lambda: clustering(bases))
    if set(groups) != set(ids):
        raise ValueError('Clustering must assign every client exactly once')
    spaces = {}
    for client_id in ids:
        group, names = groups[client_id], tuple(label_spaces[client_id])
        if group in spaces and spaces[group] != names:
            raise ValueError(f'{group}: incompatible ordered label spaces; baseline GAN averaging '
                             'cannot merge them. Choose a compatible clustering strategy.')
        spaces[group] = names
    truth = ByClassMapping(spaces)()
    mapper = ByClassMapping(spaces) if mapping_strategy == 'by_class' else mapping_strategy
    if not callable(mapper):
        raise ValueError('mapping_strategy must be callable or by_class')
    counts = {group: len(names) for group, names in spaces.items()}
    tests = [(name, groups[client_id], loader) for name, client_id, loader in test_sets]
    mapping = model = None
    history = []
    for round_index in range(config.rounds):
        # First aggregation always follows local training of all participants.
        selected = clients if round_index == 0 else rng.sample(
            clients, max(1, int(len(clients) * config.sample_fraction)))
        states = server.snapshot()
        down_bytes = 0
        if round_index:
            for client in selected:
                state = states[groups[client.id]]
                client.receive(state)
                down_bytes += payload_bytes(state)
        def train_local():
            if round_index == 0 and config.pretrained:
                return {}
            stats = {}
            for client in selected:
                seed_stage(f'client:{client.id}:round:{round_index}')
                stats[client.id] = client.train(round_index, train_classifier=round_index < config.mapping_round)
            return stats
        local_stats, local_seconds = timed(train_local)
        messages = [c.export(groups[c.id]) for c in selected]
        _, aggregate_seconds = timed(lambda: server.aggregate(messages))
        row = dict(round=round_index + 1, selected_clients=[c.id for c in selected],
                   local=local_stats, seconds={'local_training': local_seconds,
                                               'aggregation': aggregate_seconds},
                   gan_upload_bytes=sum(payload_bytes(m.gan) for m in messages),
                   gan_download_bytes=down_bytes, classifier_upload_bytes=0)
        if round_index == 0:
            row['seconds'].update(basis=basis_seconds, clustering=cluster_seconds)
            row['basis_upload_bytes'] = sum(b.nbytes for b in bases.values())
        if round_index + 1 >= config.mapping_round:
            seed_stage(f'generators:{round_index}')
            def materialize_generators():
                result = {}
                for group, state in server.snapshot().items():
                    generator = generator_factory(counts[group]).to(config.device)
                    generator.load_state_dict(state.generator)
                    result[group] = generator.eval()
                return result
            generators, seconds = timed(materialize_generators)
            row['seconds']['generator_materialization'] = seconds
            if mapping is None:
                classifiers = defaultdict(list)
                if mapping_strategy != 'by_class' and getattr(mapper, 'needs_classifiers', True):
                    for client in clients:
                        snapshot = client.classifier_snapshot()
                        row['classifier_upload_bytes'] += sum(
                            t.numel() * t.element_size() for t in snapshot.state_dict().values())
                        classifiers[groups[client.id]].append(snapshot)
                inputs = MappingInputs(generators, dict(classifiers), counts, spaces)
                seed_stage('mapping')
                mapping, seconds = timed(lambda: validate_mapping(mapper(inputs), counts))
                row['seconds']['mapping'] = seconds
                del inputs, classifiers
            row['mapping_metrics'] = mapping_metrics(mapping, truth)
            seed_stage(f'global_training:{round_index}')
            model, seconds = timed(lambda: global_trainer(generators, mapping))
            row['seconds']['global_training'] = seconds
            row['evaluation'], seconds = timed(
                lambda: evaluator(model, tests, mapping, truth, config.device))
            row['seconds']['evaluation'] = seconds
            del generators
        history.append(row)
        if record:
            record(row)
    return dict(protocol='image_bi', groups=groups, mapping=mapping,
                ground_truth=truth, model=model, history=history)


def main():
    from setup import run_cli
    run_cli(run)


if __name__ == '__main__':
    main()
