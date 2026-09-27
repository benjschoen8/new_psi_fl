"""Shared-trunk conditional GAN keyed by label codes.

Condition of label L: a fixed code vector c_L (secfl.label_codes: from the public dictionary
id, identical for every holder). Nothing per label is learned:
  - G (the trunk) takes [z, c_L]; it is the only aggregated model (FedAvg through SecAgg),
  - D stays on the client, conditioned on the client's own local label indices,
  - c_L is fixed, so label 1 held by clients A and B uses the same condition and A's and
    B's trunk updates for label 1 add up naturally.
"""
import numpy as np
import torch
from torch import nn

from nets import DCGANGenerator
from .secagg_primitives import prg
from .settle import flatten, unflatten


def code_vector(seed: bytes, dim: int) -> np.ndarray:
    """Deterministic +-1/sqrt(dim) vector from a 32-byte seed (ChaCha20): unit norm, and
    codes of different labels are nearly orthogonal."""
    bits = prg(seed, dim, 64) >> np.uint64(63)
    return ((bits.astype(np.float32) * 2 - 1) / np.sqrt(dim)).astype(np.float32)


class CodeDCGANGenerator(DCGANGenerator):
    """DCGANGenerator whose condition is a code vector instead of a one-hot label."""

    def __init__(self, code_dim=128, noise_dim=128, img_size=32, channels=3):
        super().__init__(code_dim, noise_dim, img_size, channels)
        self.code_dim = code_dim

    def forward(self, z, codes):
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        return self.net(torch.cat([z, codes.view(codes.size(0), self.code_dim, 1, 1)], dim=1))


def generator_state(g: nn.Module) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in g.state_dict().items()}


def load_generator_state(g: nn.Module, state: dict):
    g.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})


class ClientCodeGAN:
    """One client: shared trunk G (synced with the global one) + private local D."""

    def __init__(self, codes: dict, generator: nn.Module, discriminator: nn.Module, config=None, device='cpu',
                 seed=None):
        """codes {local label index: code vector}; discriminator conditions on local indices.

        seed: private noise RNG, so clients trained in parallel threads stay reproducible.
        """
        config = config or {}
        if sorted(codes) != list(range(len(codes))):
            raise ValueError('codes must cover local labels 0..k-1')
        self.device, self.G, self.D = device, generator.to(device), discriminator.to(device)
        self.C = torch.as_tensor(np.stack([codes[a] for a in range(len(codes))])).to(device)
        self.noise_dim = config.get('gen_noise_dim', 128)
        self.epochs = config.get('gen_local_epochs', 1)
        args = dict(lr=config.get('gen_lr', 2e-4), betas=(config.get('gan_beta1', .5), config.get('gan_beta2', .999)))
        self.g_opt = torch.optim.Adam(self.G.parameters(), **args)
        self.d_opt = torch.optim.Adam(self.D.parameters(), **args)
        self._ref = None
        # CUDA keeps its own generator; for MPS/CPU draw on CPU and move (portable, reproducible)
        self._rng_device = 'cuda' if str(device).startswith('cuda') else 'cpu'
        self.rng = torch.Generator(device=device if self._rng_device == 'cuda' else 'cpu')
        if seed is not None:
            self.rng.manual_seed(seed)

    def _noise(self, n):
        z = torch.randn(n, self.noise_dim, device=self._rng_device, generator=self.rng)
        return z.to(self.device)

    def state_dict(self) -> dict:
        """Everything needed to resume this client exactly (weights, optimizers, reference, RNG)."""
        return dict(G=self.G.state_dict(), D=self.D.state_dict(), g_opt=self.g_opt.state_dict(),
                    d_opt=self.d_opt.state_dict(), rng=self.rng.get_state(),
                    ref=None if self._ref is None else torch.from_numpy(self._ref))   # tensor: streamed by torch.save, no pickle copy

    def load_state_dict(self, state: dict):
        self.G.load_state_dict(state['G']); self.D.load_state_dict(state['D'])
        self.g_opt.load_state_dict(state['g_opt']); self.d_opt.load_state_dict(state['d_opt'])
        self._ref = state['ref'].numpy().copy() if torch.is_tensor(state['ref']) else state['ref']
        self.rng.set_state(state['rng'])

    def load_global(self, state: dict):
        load_generator_state(self.G, state)
        self._ref = flatten(state)[0]

    def train(self, loader) -> int:
        bce, n = nn.BCEWithLogitsLoss(), 0
        self.G.train(); self.D.train()
        for epoch in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if len(x) < 2:
                    continue
                n += len(x) if epoch == 0 else 0
                codes, real, fake = self.C[y], torch.ones(len(x), 1, device=self.device), torch.zeros(len(x), 1, device=self.device)
                self.d_opt.zero_grad()
                z = self._noise(len(x))
                d_loss = .5 * (bce(self.D(x, y).view(-1, 1), real)
                               + bce(self.D(self.G(z, codes).detach(), y).view(-1, 1), fake))
                d_loss.backward(); self.d_opt.step()
                self.g_opt.zero_grad()
                z = self._noise(len(x))
                bce(self.D(self.G(z, codes), y).view(-1, 1), real).backward()
                self.g_opt.step()
        return n

    def update(self):
        """-(theta_local - theta_global): settle(lr=1) over the sum gives weighted FedAvg."""
        local = flatten(generator_state(self.G))[0]
        return -(local - (self._ref if self._ref is not None else np.zeros_like(local)))


def classifier_inputs(generator: nn.Module, codes: list):
    """Adapter for training.GlobalClassifierTrainer: class k is sampled with code k."""
    C = torch.as_tensor(np.stack(codes))

    class _Coded(nn.Module):
        def __init__(self, g, k):
            super().__init__()
            self.g, self.k = g, k

        def forward(self, z, y):
            return self.g(z, C[self.k].to(z.device).expand(len(z), -1))

    return {str(k): _Coded(generator, k) for k in range(len(codes))}, {str(k): {0: k} for k in range(len(codes))}
