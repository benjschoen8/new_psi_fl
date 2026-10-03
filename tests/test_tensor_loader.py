import unittest

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from secure_cbn import data_hash, train_guide
from tensor_loader import TensorLoader


class TensorLoaderTests(unittest.TestCase):
    def test_same_batches_hash_guide_and_global_rng_as_the_dataloader(self):
        for gray in (True, False):
            tf = [transforms.Grayscale(3)] if gray else []
            ds = datasets.FakeData(300, (3, 32, 32), 5, transform=transforms.Compose(
                tf + [transforms.ToTensor(), transforms.Normalize((.5,), (.5,))]))
            old = DataLoader(ds, batch_size=64, shuffle=True)
            new = TensorLoader(old, shuffle=True)
            self.assertEqual(new.dataset.x.shape[1], 1 if gray else 3)          # grey kept as one channel
            out = []
            for L in (old, new):
                torch.manual_seed(0)
                L.sampler.generator = torch.Generator().manual_seed(5)
                m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 5))
                acc = train_guide(m, L, 2, 1e-3, 'cpu', 9)
                batches = [b for _ in range(2) for b in L]
                out.append((acc, [p.detach().clone() for p in m.parameters()], data_hash(L), batches, torch.rand(3)))
            (a1, p1, h1, b1, g1), (a2, p2, h2, b2, g2) = out
            self.assertEqual((a1, h1), (a2, h2))
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(p1, p2)))
            self.assertTrue(all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(b1, b2)))
            torch.testing.assert_close(g1, g2, rtol=0, atol=0)                  # global RNG untouched


class BalancedBatchTests(unittest.TestCase):
    def test_every_batch_has_m_images_of_every_label_and_is_reproducible(self):
        y = torch.tensor([0] * 90 + [1] * 8 + [2] * 2)                      # very unbalanced
        L = TensorLoader.from_tensors(torch.arange(100, dtype=torch.uint8).view(100, 1, 1, 1).expand(-1, 1, 2, 2),
                                      y, 64, shuffle=True)
        runs = []
        for _ in range(2):
            L.sampler.generator = torch.Generator().manual_seed(3)
            runs.append(list(L.balanced(4)))
        self.assertEqual(len(runs[0]), 9)                                    # ceil(100 / (4 * 3))
        for (x, yb), (x2, _) in zip(*runs):
            self.assertEqual(torch.bincount(yb).tolist(), [4, 4, 4])
            self.assertTrue(torch.equal(x, x2))
            ids = torch.round((x[:, 0, 0, 0] * .5 + .5) * 255).long()      # image value = its index
            self.assertTrue(torch.equal(y[ids], yb))                         # each image has its label

    def test_per_label_generator_trains_on_balanced_batches(self):
        from nets import DCGANDiscriminator
        from secfl.cbn_gan import ClientCBNGAN, DCGANTemplate, PerLabelGenerator
        y = torch.tensor([0] * 50 + [1] * 6 + [2] * 4)
        L = TensorLoader.from_tensors(torch.zeros(60, 1, 32, 32, dtype=torch.uint8), y, 64, shuffle=True)
        seen = []
        orig = L.balanced
        L.balanced = lambda m: (seen.append(torch.bincount(b[1]).tolist()) or b for b in orig(m))
        g = ClientCBNGAN(PerLabelGenerator(3, DCGANTemplate(8, 3, (8, 4, 2))), DCGANDiscriminator(3),
                         dict(gen_noise_dim=8, gen_label_batch=8), 'cpu', seed=0)
        self.assertEqual(g.train(L), {0: 50, 1: 6, 2: 4})                    # real counts (aggregation)
        self.assertEqual(seen, [[8, 8, 8]] * 3)                              # ceil(60 / 24) steps


if __name__ == '__main__':
    unittest.main()
