import secrets
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nets import DCGANDiscriminator
from secfl.code_gan import (code_vector, CodeDCGANGenerator, ClientCodeGAN, generator_state,
                            load_generator_state, classifier_inputs)
from secfl.settle import flatten, unflatten, settle
from training import GlobalClassifierTrainer

CFG = dict(gen_noise_dim=8, gen_local_epochs=1)


def gen():
    torch.manual_seed(123)                   # public seed: every client starts from the same trunk
    return CodeDCGANGenerator(code_dim=16, noise_dim=8)


def loader(n_labels, per=6):
    y = torch.arange(n_labels).repeat_interleave(per)
    return DataLoader(TensorDataset(torch.rand(len(y), 3, 32, 32) * 2 - 1, y), batch_size=6, shuffle=True)


class CodeGANTests(unittest.TestCase):
    def test_code_vector(self):
        s = secrets.token_bytes(32)
        a, b = code_vector(s, 128), code_vector(s, 128)
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        other = code_vector(secrets.token_bytes(32), 128)
        self.assertLess(abs(float(a @ other)), .4)                       # nearly orthogonal

    def test_generator_shapes(self):
        g = gen()
        out = g(torch.randn(3, 8), torch.as_tensor(np.stack([code_vector(secrets.token_bytes(32), 16)] * 3)))
        self.assertEqual(out.shape, (3, 3, 32, 32))

    def test_fedavg_of_trunk_across_different_label_sets(self):
        shared = code_vector(secrets.token_bytes(32), 16)                 # label 1 held by both
        a = ClientCodeGAN({0: shared, 1: code_vector(secrets.token_bytes(32), 16)}, gen(), DCGANDiscriminator(2), CFG)
        b = ClientCodeGAN({0: code_vector(secrets.token_bytes(32), 16), 1: shared}, gen(), DCGANDiscriminator(2), CFG)
        start = generator_state(gen())
        for c in (a, b):
            c.load_global(start)
        na, nb = a.train(loader(2)), b.train(loader(2, 9))
        ua, ub = a.update(), b.update()
        flat0, spec = flatten(start)
        new, _ = settle({'G': flat0}, {'G': na * ua + nb * ub}, {'G': na + nb}, {'G': 2}, lr=1, threshold=1)
        wa, wb = flatten(generator_state(a.G))[0], flatten(generator_state(b.G))[0]
        np.testing.assert_allclose(new['G'], (na * wa + nb * wb) / (na + nb), rtol=1e-5, atol=1e-6)
        g = gen(); load_generator_state(g, unflatten(new['G'], spec))  # averaged trunk loads back

    def test_classifier_from_codes(self):
        codes = [code_vector(secrets.token_bytes(32), 16) for _ in range(3)]
        trainer = GlobalClassifierTrainer(lambda k: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, k)),
                                          dict(gen_noise_dim=8, global_samples_per_class=4, global_model_epochs=1))
        model = trainer(*classifier_inputs(gen().eval(), codes))
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
