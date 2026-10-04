import unittest

import torch

from training import augment


class AugmentTests(unittest.TestCase):
    def test_off_is_identity_and_flip_mirrors(self):
        x = torch.rand(4, 3, 32, 32) * 2 - 1
        self.assertTrue(torch.allclose(augment(x, shift=0, flip=False, noise=0), x, atol=1e-6))
        torch.manual_seed(0)
        y = augment(x, shift=0, flip=True, noise=0)
        for a, b in zip(x, y):                                   # each image: kept or mirrored
            self.assertTrue(torch.allclose(b, a, atol=1e-6) or torch.allclose(b, a.flip(2), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
