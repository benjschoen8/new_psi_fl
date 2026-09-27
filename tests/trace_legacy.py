"""Execute original and revised control flow on the same deterministic CPU fixture.

Only model factories are replaced with small networks; original server.run(),
client.update(), mapping, global training and evaluation execute unchanged.
Run: python -m tests.trace_legacy
"""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'legacy'))  # legacy: reference comparisons only
import trainer.GeFL_gan_pacfl_iid.client as old_client
import trainer.GeFL_gan_pacfl_iid.server as old_server
from client import Client
from clustering import PACFL, local_basis
from aggregation import GeFLAggregation, weighted_average
from server import Server
from main import run, RunConfig
from mapping import ImageBiMapping
from evaluation import evaluate_global
from training import GlobalClassifierTrainer
from setup import seed_all


CONFIG = dict(global_rounds=3, local_epochs=1, gen_local_epochs=1, batch_size=4,
              gen_noise_dim=128, global_model_epochs=1, global_samples_per_class=4,
              entropy_ratio=.25, use_new_entropy_method=True, global_feature_dim=256,
              global_model_optim_lr=.001, channels=3, img_size=2)


class Generator(nn.Module):
    def __init__(self, num_classes, noise_dim=128, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 2)
        self.linear = nn.Linear(noise_dim + 2, 12)

    def forward(self, noise, labels):
        return self.linear(torch.cat((noise, self.embedding(labels)), 1)).tanh().reshape(-1, 3, 2, 2)


class Discriminator(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 2)
        self.linear = nn.Linear(14, 1)

    def forward(self, images, labels):
        return self.linear(torch.cat((images.flatten(1), self.embedding(labels)), 1))


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(12, num_classes)

    def forward(self, images):
        features = images.flatten(1)
        return features, self.linear(features)


class Logger:
    def __init__(self, directory):
        self.log_dir = str(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def log(self, text):
        pass


def loaders(cid):
    x = torch.full((8, 12), -1.)
    x[:, :6] = .7 if cid < 2 else -1.
    x[:, 6:] = -1. if cid < 2 else .7
    x = x.reshape(8, 3, 2, 2)
    y = torch.tensor([0, 1] * 4)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, shuffle=True), DataLoader(ds, batch_size=4)


def args():
    return SimpleNamespace(device='cpu', algorithm='GeFL_gan_pacfl_iid',
                           start_mapping_epoch=2, pacfl_basis_budget=1,
                           pacfl_cluster_alpha=20, label_mapping='image-bi')


def build_old(logger, fraction):
    clients = []
    for cid in range(4):
        train, test = loaders(cid)
        clients.append(old_client.Client(node_id=cid, args=args(), dataset_name=f'd{cid // 2}',
            train_loader=train, test_loader=test, model=Classifier(2), class_name_set=['0', '1'],
            model_name='tiny', logger=logger, **CONFIG))
    config = dict(CONFIG, sample_frac=fraction)
    server = old_server.Server(clients=clients, node_id=-1, args=args(), dataset_name=None,
        train_loader=None, test_loader=None, model=None, class_name_set=None,
        model_name='tiny', logger=logger, exp_conf=config, **config)
    return clients, server


def build_new():
    clients, tests = [], []
    for cid in range(4):
        train, test = loaders(cid)
        clients.append(Client(cid, Classifier(2), Generator(2), Discriminator(2), train, 2, CONFIG))
        tests.append((f'd{cid // 2}', cid, test))
    return clients, tests


def states(clients):
    return {c.id: {name: {k: t.detach().cpu().clone() for k, t in getattr(c, name).state_dict().items()}
                   for name in ('model', 'generator', 'discriminator')} for c in clients}


def difference(a, b):
    deltas = [float((a[c][name][key] - value).abs().max())
              for c, models in b.items() for name, values in models.items() for key, value in values.items()]
    return dict(exactly_equal=all(d == 0 for d in deltas), max_abs_difference=max(deltas))


def execute_pair(root, fraction):
    logger = Logger(root / f'fraction_{fraction}')
    old_rounds, new_rounds, old_events, new_events = [], [], [], []
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(old_client, 'DCGANGenerator', Generator))
        stack.enter_context(patch.object(old_client, 'DCGANDiscriminator', Discriminator))
        stack.enter_context(patch.object(old_server, 'DCGANGenerator', Generator))
        stack.enter_context(patch.object(old_server, 'ResNet', lambda *a, **kw: Classifier(kw['num_classes'])))
        stack.enter_context(patch.object(old_server, 'tqdm', lambda values, **kw: values))
        seed_all(9)
        original_clients, original = build_old(logger, fraction)
        original_initial = states(original_clients)
        select, distribute, update, aggregate = original.sample_clients, original.distribute_model, original.local_update, original.aggregate
        def selected():
            select()
            old_events.append(dict(event='select', round=original.glob_iter + 1,
                                   ids=[c.id for c in original.selected_clients]))
        def distributed():
            old_events.append(dict(event='distribute', round=original.glob_iter + 1))
            distribute()
        def updated():
            old_events.append(dict(event='local_train', round=original.glob_iter + 1,
                                   classifier=original.glob_iter + 1 <= 2))
            update()
        def aggregated():
            old_events.append(dict(event='aggregate', round=original.glob_iter + 1))
            aggregate()
            old_rounds.append(states(original_clients))
        original.sample_clients, original.distribute_model = selected, distributed
        original.local_update, original.aggregate = updated, aggregated
        for attr, event in [('train_global_inference_model', 'global_train'), ('test_global_inference_model', 'evaluate')]:
            method = getattr(original, attr)
            def wrapper(method=method, event=event):
                old_events.append(dict(event=event, round=original.glob_iter + 1))
                return method()
            setattr(original, attr, wrapper)
        mapper = old_server.label_mapping
        def trace_mapping(**kwargs):
            old_events.append(dict(event='mapping', round=original.glob_iter + 1))
            return mapper(**kwargs)
        stack.enter_context(patch.object(old_server, 'label_mapping', trace_mapping))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            original.run()
        original_groups = {c.id: c.group_name for c in original_clients}
        with open(Path(logger.log_dir) / 'global_model_acc_mix.csv') as stream:
            old_accuracy = [float(row['Accuracy']) / 100 for row in csv.DictReader(stream)]

    seed_all(9)
    revised_clients, tests = build_new()
    initial_comparison = difference(original_initial, states(revised_clients))
    for client in revised_clients:
        train, receive = client.train, client.receive
        def trace_train(round_index, train_classifier=True, method=train, cid=client.id):
            new_events.append(dict(event='local_train', round=round_index + 1,
                                   client_id=cid, classifier=train_classifier))
            return method(round_index, train_classifier)
        def trace_receive(state, method=receive, cid=client.id):
            new_events.append(dict(event='distribute', round=len(new_rounds) + 1, client_id=cid))
            return method(state)
        client.train, client.receive = trace_train, trace_receive
    aggregator = GeFLAggregation()
    def trace_aggregate(messages):
        new_events.append(dict(event='aggregate', round=len(new_rounds) + 1,
                               ids=[m.client_id for m in messages]))
        return aggregator(messages)
    revised_server = Server(trace_aggregate)
    strategy = ImageBiMapping(logger, noise_dim=128)
    def trace_strategy(inputs):
        new_events.append(dict(event='mapping', round=len(new_rounds) + 1))
        return strategy(inputs)
    trainer = GlobalClassifierTrainer(Classifier, CONFIG)
    def trace_trainer(generators, mapping):
        new_events.append(dict(event='global_train', round=len(new_rounds) + 1))
        return trainer(generators, mapping)
    def trace_evaluator(*values):
        new_events.append(dict(event='evaluate', round=len(new_rounds) + 1))
        return evaluate_global(*values)
    def record(row):
        new_rounds.append(states(revised_clients))
    result = run(revised_clients, revised_server, clustering=PACFL(20), mapping_strategy=trace_strategy,
        global_trainer=trace_trainer, generator_factory=Generator, label_spaces={i: ('0', '1') for i in range(4)},
        test_sets=tests, config=RunConfig(rounds=3, mapping_round=2, sample_fraction=fraction, basis_budget=1, seed=9),
        evaluator=trace_evaluator, record=record)
    summary = dict(sample_fraction=fraction, initial_models=initial_comparison,
        original_groups=original_groups, revised_groups=result['groups'],
        original_events=old_events, revised_events=new_events,
        original_selected=[e['ids'] for e in old_events if e['event']=='select'],
        revised_selected=[row['selected_clients'] for row in result['history']],
        client_models_by_round=[difference(a,b) for a,b in zip(old_rounds,new_rounds)],
        original_mapping=original.local_id_to_global_id, revised_mapping=result['mapping'],
        original_accuracy=old_accuracy,
        revised_accuracy=[row['evaluation']['accuracy'] for row in result['history'] if 'evaluation' in row])
    assert initial_comparison['exactly_equal']
    assert original_groups == result['groups']
    assert [e['round'] for e in old_events if e['event']=='mapping']==[2]
    assert [e['round'] for e in old_events if e['event']=='global_train']==[2,3]
    assert sum(e['event']=='mapping' for e in new_events)==1
    assert sum(e['event']=='global_train' for e in new_events)==2
    return summary


def pacfl_component_check(root):
    logger = Logger(root / 'pacfl')
    original = old_server.Server.__new__(old_server.Server)
    original.args, original.logger = args(), logger
    original.clients = [SimpleNamespace(id=i, train_loader=loaders(i)[0], dataset_name='d', class_name_set=['0','1'])
                        for i in range(4)]
    original.group_label_space_meta = {}
    captured = []
    actual = old_server.calculating_adjacency
    def capture(ids, bases):
        captured.extend(copy.deepcopy(bases))
        return actual(ids, bases)
    seed_all(9)
    with patch.object(old_server, 'calculating_adjacency', capture):
        original.initialize_client_groups()
    seed_all(9)
    revised = [local_basis(loaders(i)[0], budget=1) for i in range(4)]
    exact = all(np.array_equal(a,b) for a,b in zip(captured, revised))
    assert exact
    return dict(bases_exactly_equal=exact, groups={c.id:c.group_name for c in original.clients})


def aggregation_component_check():
    seed_all(8)
    weighted = [(n, {'w':torch.randn(100)}) for n in (3,7,11)]
    a = old_server.Server.aggregate_weights(None, weighted)['w']
    b = weighted_average(weighted)['w']
    delta = float((a-b).abs().max())
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    return dict(exactly_equal=torch.equal(a,b), max_abs_difference=delta, allclose_1e_6=True)


def global_training_component_check(root):
    """Hold generator outputs, model init and RNG consumption equal to isolate training math."""
    seed_all(18)
    template = Generator(2)
    mapping = {'g': {0: 0, 1: 1}}
    original = old_server.Server.__new__(old_server.Server)
    original.local_id_to_global_id = mapping
    original.model = None
    original.device = 'cpu'
    original.exp_conf = CONFIG
    original.logger = Logger(root / 'global_training')
    original.group_label_space_meta = {'g': ['0', '1']}
    original.global_gen_states = {'g': template.state_dict()}
    original.global_samples_per_class = 4
    original.global_model_epochs = 1
    original.batch_size = 4
    revised = GlobalClassifierTrainer(Classifier, CONFIG)
    with patch.object(old_server, 'DCGANGenerator', lambda **kw: copy.deepcopy(template)), \
         patch.object(old_server, 'ResNet', lambda *a, **kw: Classifier(kw['num_classes'])), \
         patch.object(old_server, 'tqdm', lambda values, **kw: values):
        seed_all(27)
        original.train_global_inference_model()
        seed_all(27)
        revised({'g': copy.deepcopy(template)}, mapping)
    equality = all(torch.equal(value, revised.model.state_dict()[key])
                   for key, value in original.model.state_dict().items())
    assert equality
    return dict(weights_exactly_equal_with_aligned_rng_and_generator_outputs=equality)


def evaluation_component_check(root):
    original = old_server.Server.__new__(old_server.Server)
    original.model = Classifier(1)
    original.device = 'cpu'
    original.glob_iter = 0
    original.logger = Logger(root / 'evaluation')
    original.local_id_to_global_id = {'g': {0: 0, 1: 0}}
    loader = loaders(0)[1]
    original.clients = [SimpleNamespace(dataset_name='d', group_name='g', test_loader=loader)]
    original.test_global_inference_model()
    with open(Path(original.logger.log_dir) / 'global_model_acc_mix.csv') as stream:
        accuracy = float(list(csv.DictReader(stream))[-1]['Accuracy']) / 100
    revised = evaluate_global(original.model, [('d','g',loader)], original.local_id_to_global_id,
                              {'g': {0: 0, 1: 1}})
    assert accuracy == 1. and revised['accuracy'] == 0.
    return dict(same_model_and_data_wrong_merge_original_accuracy=accuracy,
                same_model_and_data_wrong_merge_revised_accuracy=revised['accuracy'])


def main():
    torch.set_num_threads(1)
    import tempfile
    Path('runs').mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='legacy_trace_',dir='runs'))
    result=dict(pacfl=pacfl_component_check(root), aggregation=aggregation_component_check(),
                global_training=global_training_component_check(root),
                evaluation=evaluation_component_check(root),
                end_to_end=[execute_pair(root,fraction) for fraction in (1., .5)])
    (root/'trace.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print(f'Trace written: {root}/trace.json')


if __name__=='__main__':
    main()
