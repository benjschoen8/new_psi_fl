import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'legacy'))  # legacy: reference comparisons only
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from aggregation import GeFLAggregation
from evaluation import evaluate_global
from main import run, RunConfig
from mapping import ImageBiMapping, ByClassMapping
from server import Server
from setup import seed_all, label_names
from smoke import build_smoke_clients
from training import GlobalClassifierTrainer


CONFIG = dict(gen_noise_dim=4, gen_local_epochs=1, local_epochs=1,
              global_samples_per_class=4, global_model_epochs=1, batch_size=4)


class QuietLogger:
    def log(self, text):
        pass


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def execute(self, strategy, pretrained=False, clustering=lambda bases: {i: str(i) for i in bases}):
        seed_all(9)
        clients, spaces, tests, _, gen_factory, classifier_factory = build_smoke_clients(CONFIG)
        events = []
        for c in clients:
            original = c.train
            def train(round_index, train_classifier=True, original=original, cid=c.id):
                events.append(('train', cid, round_index))
                return original(round_index, train_classifier)
            c.train = train
        class AuditedAggregation(GeFLAggregation):
            def __call__(self, messages):
                events.append(('aggregate',))
                return super().__call__(messages)
        server = Server(AuditedAggregation())
        result = run(clients, server, clustering=clustering,
                     mapping_strategy=strategy, global_trainer=GlobalClassifierTrainer(classifier_factory, CONFIG),
                     generator_factory=gen_factory, label_spaces=spaces, test_sets=tests,
                     config=RunConfig(rounds=2, mapping_round=1, seed=9, pretrained=pretrained))
        return result, clients, server, events

    def test_real_training_and_order_and_server_snapshot(self):
        result, clients, server, events = self.execute('by_class')
        self.assertEqual(events[:3], [('train', 0, 0), ('train', 1, 0), ('aggregate',)])
        self.assertEqual(result['history'][0]['gan_download_bytes'], 0)
        self.assertGreater(result['history'][1]['gan_download_bytes'], 0)
        self.assertEqual(result['history'][0]['evaluation']['samples'], 16)
        self.assertIn('old_acc', result['history'][0]['evaluation'])
        self.assertIn('ground_truth_acc', result['history'][0]['evaluation'])
        self.assertEqual(result['history'][0]['mapping_metrics']['f1'], 1.)
        snapshot = server.snapshot()
        key = next(iter(snapshot['0'].generator))
        snapshot['0'].generator[key].fill_(999)
        self.assertFalse(torch.all(server.snapshot()['0'].generator[key] == 999))
        self.assertFalse(hasattr(server, 'clients'))

    def test_no_clustering_keeps_one_generator_per_client(self):
        result, _, server, _ = self.execute('by_class', clustering=None)
        self.assertEqual(result['groups'], {0: 'Client_0', 1: 'Client_1'})
        self.assertEqual(set(server.snapshot()), {'Client_0', 'Client_1'})
        self.assertEqual(result['history'][0]['basis_upload_bytes'], 0)

    def test_psi_trivial_circuit_matches_identical_descriptions(self):
        from mapping import PSITrivialCircuit
        result, _, _, _ = self.execute(PSITrivialCircuit())
        self.assertEqual(result['mapping'], {'0': {0: 0, 1: 1}, '1': {0: 0, 1: 1}})
        self.assertEqual(result['history'][0]['classifier_upload_bytes'], 0)
        other_language, _, _, _ = self.execute(PSITrivialCircuit(('en', 'zh')))
        self.assertEqual(len({g for m in other_language['mapping'].values() for g in m.values()}), 4)

    def test_fuzzy_psi_circuit_executes(self):
        from mapping import FuzzyPSICircuit
        for method in ('filter', 'affscan'):
            result, _, _, _ = self.execute(FuzzyPSICircuit(method, noise_dim=4, samples=4))
            self.assertEqual({k: set(v) for k, v in result['mapping'].items()}, {'0': {0, 1}, '1': {0, 1}})
            self.assertEqual(result['history'][0]['classifier_upload_bytes'], 0)

    def test_with_key_variants_release_keys_to_holders_only(self):
        from mapping import PSITrivialCircuitWithKey, FuzzyPSICircuitWithKey
        for strategy in (PSITrivialCircuitWithKey(('en', 'zh')),
                         FuzzyPSICircuitWithKey('filter', noise_dim=4, samples=4)):
            result, _, _, _ = self.execute(strategy, clustering=None)
            held = {g: {strategy.slots[i] for i in m.values()} for g, m in result['mapping'].items()}
            self.assertEqual(len(strategy.keys), strategy.max_labels)       # every slot keyed
            for group, keys in strategy.client_keys.items():
                self.assertEqual(set(keys), held[group])
                for L, k in keys.items():
                    self.assertEqual(k, strategy.keys[L])
            self.assertEqual(result['history'][0]['classifier_upload_bytes'], 0)

    def test_cross_group_image_bi_executes(self):
        result, _, _, _ = self.execute(ImageBiMapping(QuietLogger(), noise_dim=4, samples=4))
        self.assertEqual(set(result['mapping']), {'0', '1'})
        self.assertEqual(set(result['mapping']['0']), {0, 1})
        self.assertEqual(len(result['history']), 2)

    def test_strategy_rng_cannot_change_client_training(self):
        _, reference_clients, _, _ = self.execute('by_class')
        def noisy_mapping(inputs):
            torch.randn(137)
            return ByClassMapping({key: ('zero', 'one') for key in inputs.num_classes})()
        _, trial_clients, _, _ = self.execute(noisy_mapping)
        for a, b in zip(reference_clients, trial_clients):
            for key, value in a.generator.state_dict().items():
                self.assertTrue(torch.equal(value, b.generator.state_dict()[key]), key)

    def test_pretrained_flag_rejects_random_models(self):
        with self.assertRaisesRegex(ValueError, 'trained client'):
            self.execute('by_class', pretrained=True)

    def test_truth_follows_wrapped_permutation(self):
        from types import SimpleNamespace
        base = SimpleNamespace(classes=['A', 'a', 'B'])
        wrapped = SimpleNamespace(dataset=base, mapping_dict={0: 2, 1: 0, 2: 1})
        subset = SimpleNamespace(dataset=wrapped)
        self.assertEqual(label_names(subset, 'EMNIST'), ('a', 'B', 'A'))

    def test_merged_class_counts_all_test_samples_as_wrong(self):
        model = torch.nn.Linear(2, 1)
        loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
        score = evaluate_global(model, [('d', 'g', loader)], {'g': {0: 0, 1: 0}}, {'g': {0: 3, 1: 4}})
        self.assertEqual(score['samples'], 4)
        self.assertEqual(score['accuracy'], 0.)
        self.assertEqual(score['ground_truth_acc'], 0.)
        self.assertEqual(score['old_acc'], 1.)
        self.assertEqual(score['old_samples'], 4)
        self.assertEqual(score['ambiguous_predictions'], 4)

    def test_old_acc_skips_unmapped_labels_without_changing_truth_denominator(self):
        model = torch.nn.Linear(2, 1)
        loader = DataLoader(TensorDataset(torch.ones(4, 2), torch.tensor([0, 1, 0, 1])), batch_size=2)
        score = evaluate_global(model, [('d', 'g', loader)], {'g': {0: 0}}, {'g': {0: 0, 1: 1}})
        self.assertEqual(score['old_acc'], 1.)
        self.assertEqual(score['old_samples'], 2)
        self.assertEqual(score['old_correct'], 2)
        self.assertEqual(score['ground_truth_acc'], .5)
        self.assertEqual(score['samples'], 4)
        self.assertEqual(score['by_dataset']['d']['old_acc'], 1.)
        self.assertEqual(score['by_dataset']['d']['ground_truth_acc'], .5)

    def test_old_acc_matches_original_evaluation_on_same_logits(self):
        import csv
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from trainer.GeFL_gan_pacfl_iid.server import Server as LegacyServer
        class IdentityClassifier(torch.nn.Module):
            def forward(self, x):
                return x, x
        model = IdentityClassifier()
        def loader(labels, predictions):
            x = torch.nn.functional.one_hot(torch.tensor(predictions), 2).float()
            return DataLoader(TensorDataset(x, torch.tensor(labels)), batch_size=2)
        tests = [('a', 'g1', loader([0, 1, 2], [0, 0, 1])),
                 ('b', 'g2', loader([0, 1, 1, 0], [1, 1, 0, 1])),
                 ('b', 'missing', loader([0], [1]))]
        table = {'g1': {0: 0, 1: 1}, 'g2': {0: 1, 1: 0}}
        truth = {'g1': {0: 0, 1: 1, 2: 2}, 'g2': {0: 1, 1: 0}, 'missing': {0: 1}}
        with tempfile.TemporaryDirectory() as directory:
            old = LegacyServer.__new__(LegacyServer)
            old.model, old.device, old.glob_iter = model, 'cpu', 0
            old.logger = SimpleNamespace(log_dir=directory, log=lambda text: None)
            old.local_id_to_global_id = table
            old.clients = [SimpleNamespace(dataset_name=name, group_name=group, test_loader=ds)
                           for name, group, ds in tests]
            old.test_global_inference_model()
            current = evaluate_global(model, tests, table, truth)
            def csv_accuracy(name):
                with (Path(directory) / f'global_model_acc_{name}.csv').open() as stream:
                    return float(next(csv.DictReader(stream))['Accuracy'])
            self.assertEqual(round(current['old_acc'] * 100, 2), csv_accuracy('mix'))
            for name in ('a', 'b'):
                self.assertEqual(round(current['by_dataset'][name]['old_acc'] * 100, 2), csv_accuracy(name))
        self.assertEqual(current['old_correct'], 4)
        self.assertEqual(current['old_samples'], 6)
        self.assertEqual(current['samples'], 8)

    def test_old_acc_empty_mapping_has_zero_denominator(self):
        from evaluation import old_acc
        self.assertEqual(old_acc([0, 1], [0, 1], {}),
                         dict(accuracy=0., correct=0, samples=0))

    def test_image_bi_matches_permuted_labels_and_rejects_failed_cycle(self):
        from contracts import MappingInputs
        class Generator(torch.nn.Module):
            def __init__(self, reverse=False):
                super().__init__()
                self.reverse = reverse
            def forward(self, noise, labels):
                return torch.nn.functional.one_hot(1 - labels if self.reverse else labels, 2).float()
        class Classifier(torch.nn.Module):
            def __init__(self, reverse=False, constant=False):
                super().__init__()
                self.reverse, self.constant = reverse, constant
            def forward(self, images):
                logits = images.flip(1) if self.reverse else images
                if self.constant:
                    logits = torch.tensor([[1., 0.]]).expand(len(images), -1)
                return images, 30 * logits
        inputs = MappingInputs({'a': Generator(), 'b': Generator(True)},
                               {'a': [Classifier()], 'b': [Classifier(True)]}, {'a': 2, 'b': 2})
        for use_new in (True, False):
            mapper = ImageBiMapping(QuietLogger(), noise_dim=4, samples=4, use_new_entropy=use_new)
            table = mapper(inputs)
            self.assertEqual(table['a'][0], table['b'][1])
            self.assertEqual(table['a'][1], table['b'][0])
            self.assertNotEqual(table['a'][0], table['a'][1])
            inconsistent = MappingInputs(inputs.generators,
                {'a': [Classifier(constant=True)], 'b': [Classifier(True)]}, inputs.num_classes)
            table = mapper(inconsistent)
            self.assertNotEqual(table['a'][1], table['b'][0])

    def test_actual_dcgan_training_matches_legacy_local_methods(self):
        import copy
        from types import MethodType
        from nets import DCGANGenerator, DCGANDiscriminator
        from trainer.GeFL_gan_pacfl_iid.client import Client as LegacyClient
        from client import Client
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2))
        loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
        client = Client(0, model, DCGANGenerator(2, noise_dim=4), DCGANDiscriminator(2), loader, 2, CONFIG)
        legacy = copy.deepcopy(client)
        legacy.train_generator = MethodType(LegacyClient.train_generator, legacy)
        legacy.train_target_model = MethodType(LegacyClient.train_target_model, legacy)
        seed_all(33)
        legacy.train_generator()
        legacy.train_target_model()
        seed_all(33)
        client.train(0)
        for model_name in ('generator', 'discriminator', 'model'):
            for key, value in getattr(client, model_name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(legacy, model_name).state_dict()[key]),
                                f'{model_name}.{key}')

    def test_unsampled_groups_are_retained(self):
        from contracts import GANState, ClientUpdate
        server = Server(GeFLAggregation())
        def msg(cid, group, value):
            return ClientUpdate(cid, group, 2, GANState({'w': torch.tensor([value])}, {'w': torch.tensor([value])}))
        server.aggregate([msg(0, 'a', 1.), msg(1, 'b', 2.)])
        server.aggregate([msg(0, 'a', 3.)])
        self.assertEqual(server.snapshot()['a'].generator['w'].item(), 3.)
        self.assertEqual(server.snapshot()['b'].generator['w'].item(), 2.)

    def test_shipped_optimizer_settings_are_consumed(self):
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.load('config.yaml'))
        self.assertIn('gan_beta1', config)
        self.assertIn('gan_beta2', config)
        config.update(CONFIG, gan_beta1=.3, gan_beta2=.8)
        client = build_smoke_clients(config)[0][0]
        self.assertEqual(client.g_optimizer.param_groups[0]['betas'], (.3, .8))


if __name__ == '__main__':
    unittest.main()
