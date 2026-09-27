import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nets import DCGANGenerator, DCGANDiscriminator
from secfl.label_gan import ClientLabelGANs, pair_state, as_trainer_inputs, unflatten_pair
from secfl.settle import flatten, settle
from training import GlobalClassifierTrainer

CONFIG = dict(gen_noise_dim=8, gen_local_epochs=1)
G = lambda: DCGANGenerator(1, noise_dim=8)
D = lambda: DCGANDiscriminator(1)


def loader(labels, n_per=6):
    torch.manual_seed(0)
    y = torch.tensor([l for l in labels for _ in range(n_per)])
    return DataLoader(TensorDataset(torch.randn(len(y), 3, 32, 32).clamp(-1, 1), y), batch_size=8, shuffle=True)


class LabelGANTests(unittest.TestCase):
    def test_train_counts_and_only_own_label_moves(self):
        c = ClientLabelGANs({0: 5, 1: 7}, G, D, CONFIG)
        before = {L: pair_state(p['G'], p['D']) for L, p in c.pairs.items()}
        counts = c.train(loader([1], 10))                     # only local label 1 (= global 7) present
        self.assertEqual(counts, {5: 0, 7: 10})
        for k, v in pair_state(c.pairs[5]['G'], c.pairs[5]['D']).items():
            np.testing.assert_array_equal(v, before[5][k])
        self.assertTrue(any(not np.array_equal(v, before[7][k])
                            for k, v in pair_state(c.pairs[7]['G'], c.pairs[7]['D']).items()))
        grads, used = c.updates(counts)
        self.assertEqual(set(grads), {7})                     # label without samples not uploaded

    def test_settle_lr1_is_weighted_fedavg_of_pairs(self):
        a, b = ClientLabelGANs({0: 3}, G, D, CONFIG), ClientLabelGANs({2: 3}, G, D, CONFIG)
        start = pair_state(a.pairs[3]['G'], a.pairs[3]['D'])
        for c in (a, b):
            c.load_global({3: start})
        ca, cb = a.train(loader([0], 4)), b.train(loader([2], 12))
        (ga, na), (gb, nb) = a.updates(ca), b.updates(cb)
        flat0, _ = flatten(start)
        new, updated = settle({3: flat0}, {3: na[3] * ga[3] + nb[3] * gb[3]}, {3: na[3] + nb[3]},
                              {3: 2}, lr=1, threshold=2)
        wa = flatten(pair_state(a.pairs[3]['G'], a.pairs[3]['D']))[0]
        wb = flatten(pair_state(b.pairs[3]['G'], b.pairs[3]['D']))[0]
        np.testing.assert_allclose(new[3], (na[3] * wa + nb[3] * wb) / (na[3] + nb[3]), rtol=1e-6, atol=1e-6)
        back = unflatten_pair(new[3], start)
        self.assertEqual(set(back), set(start))

    def test_global_classifier_from_label_generators(self):
        gens = {0: G().eval(), 1: G().eval(), 2: G().eval()}
        trainer = GlobalClassifierTrainer(lambda n: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, n)),
                                          dict(gen_noise_dim=8, global_samples_per_class=4, global_model_epochs=1))
        model = trainer(*as_trainer_inputs(gens))
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 3))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ClientLabelGANs({0: 1, 1: 1}, G, D, CONFIG)


if __name__ == '__main__':
    unittest.main()
