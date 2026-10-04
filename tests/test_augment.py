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

    def test_gan_options_spectral_norm_copyable_and_diffaug_differentiable(self):
        import copy
        import io
        from nets import DCGANDiscriminator
        from secfl.cbn_gan import diff_augment, spectral_discriminator
        D = spectral_discriminator(DCGANDiscriminator(3))
        x, y = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 0])
        D(x, y)                                                    # after a forward: still copyable
        D2 = spectral_discriminator(copy.deepcopy(D))              # worker copy, wrapped again: no-op
        self.assertEqual(sum(type(m).__name__ == '_SpectralNorm' for m in D2.modules()), 4)
        buf = io.BytesIO()
        torch.save(D2, buf)
        x.requires_grad_()
        diff_augment(x).sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
