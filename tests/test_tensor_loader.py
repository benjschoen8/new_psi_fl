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


if __name__ == '__main__':
    unittest.main()
