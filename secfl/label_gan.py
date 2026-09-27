"""One GAN per global label (GeFL with per-label units).

Client side: after alignment the client knows {local label -> global L}. It keeps a
(G_L, D_L) pair for each L it holds, trains each pair only on its own samples of L,
and reports -(theta_local - theta_global) plus n_{i,L} per label (settle.py with lr=1
turns the sum into sample-weighted FedAvg).
Aggregator side: per-label generators feed the existing GlobalClassifierTrainer via
as_trainer_inputs(), treating each label as a one-class group.

Generators are conditional with one class (label input always 0), so the existing
DCGANGenerator/DCGANDiscriminator(num_classes=1) work unchanged.
"""
import copy

import numpy as np
import torch

from .settle import flatten, unflatten


def pair_state(generator, discriminator, include_d=True) -> dict:
    """G (and D) of one label as a single numpy state (the unit that is encrypted/aggregated).
    include_d=False: only G is shared; every client keeps its own D (smaller uploads)."""
    state = {f'G.{k}': v.detach().cpu().numpy().copy() for k, v in generator.state_dict().items()}
    if include_d:
        state.update({f'D.{k}': v.detach().cpu().numpy().copy() for k, v in discriminator.state_dict().items()})
    return state


def load_pair_state(generator, discriminator, state: dict):
    generator.load_state_dict({k[2:]: torch.as_tensor(v) for k, v in state.items() if k.startswith('G.')})
    d = {k[2:]: torch.as_tensor(v) for k, v in state.items() if k.startswith('D.')}
    if d:                                       # G-only states leave the local D untouched
        discriminator.load_state_dict(d)


class ClientLabelGANs:
    def __init__(self, local_to_global: dict, generator_factory, discriminator_factory, config=None, device='cpu',
                 share_d=True):
        """local_to_global {local label: global L}; factories build one-class G / D.
        share_d=False: updates carry G only; D stays local."""
        self.share_d = share_d
        config = config or {}
        if len(set(local_to_global.values())) != len(local_to_global):
            raise ValueError('local labels must map to distinct global labels')
        self.local_to_global, self.device = dict(local_to_global), device
        self.noise_dim = config.get('gen_noise_dim', 128)
        self.epochs = config.get('gen_local_epochs', 5)
        args = dict(lr=config.get('gen_lr', 2e-4), betas=(config.get('gan_beta1', .5), config.get('gan_beta2', .999)))
        self.pairs = {}
        for L in self.local_to_global.values():
            G, D = generator_factory().to(device), discriminator_factory().to(device)
            self.pairs[L] = dict(G=G, D=D, g_opt=torch.optim.Adam(G.parameters(), **args),
                                 d_opt=torch.optim.Adam(D.parameters(), **args))
        self._global = {}

    @property
    def labels(self):
        return sorted(self.pairs, key=str)

    def load_global(self, states: dict):
        """states {L: pair state} from the broadcast; labels not given keep local weights."""
        for L, state in states.items():
            if L in self.pairs:
                load_pair_state(self.pairs[L]['G'], self.pairs[L]['D'], state)
                self._global[L] = copy.deepcopy(state)

    def train(self, loader):
        """Train each label's pair on its own samples. Returns {L: n_{i,L}} (samples seen per epoch)."""
        bce = torch.nn.BCEWithLogitsLoss()
        counts = {L: 0 for L in self.pairs}
        for p in self.pairs.values():
            p['G'].train(); p['D'].train()
        for epoch in range(self.epochs):
            for x, y in loader:
                for local, L in self.local_to_global.items():
                    xs = x[y == local].to(self.device)
                    n = len(xs)
                    if n < 2:                      # BatchNorm needs >1 sample
                        continue
                    if epoch == 0:
                        counts[L] += n
                    p, zeros = self.pairs[L], torch.zeros(n, dtype=torch.long, device=self.device)
                    real, fake = torch.ones(n, 1, device=self.device), torch.zeros(n, 1, device=self.device)
                    p['d_opt'].zero_grad()
                    z = torch.randn(n, self.noise_dim, device=self.device)
                    d_loss = .5 * (bce(p['D'](xs, zeros).view(-1, 1), real)
                                   + bce(p['D'](p['G'](z, zeros).detach(), zeros).view(-1, 1), fake))
                    d_loss.backward(); p['d_opt'].step()
                    p['g_opt'].zero_grad()
                    z = torch.randn(n, self.noise_dim, device=self.device)
                    bce(p['D'](p['G'](z, zeros), zeros).view(-1, 1), real).backward()
                    p['g_opt'].step()
        return counts

    def updates(self, counts: dict):
        """{L: flat -(theta_local - theta_global)} for held labels with n > 0, plus counts.

        Before the first broadcast the reference is zero, so the "update" is -theta_local
        and settle(lr=1) from a zero start yields the weighted average of local weights.
        """
        grads, used = {}, {}
        for L in self.labels:
            if counts.get(L, 0) < 1:
                continue
            local, _ = flatten(pair_state(self.pairs[L]['G'], self.pairs[L]['D'], self.share_d))
            ref = flatten(self._global[L])[0] if L in self._global else np.zeros_like(local)
            grads[L], used[L] = -(local - ref), counts[L]
        return grads, used


def as_trainer_inputs(generators: dict):
    """{L: one-class generator} -> (generators, mapping) for training.GlobalClassifierTrainer.

    Each label becomes a group whose only local label 0 maps to global id L (L must be int).
    """
    class _OneClass(torch.nn.Module):          # trainer calls G(z, y) with y = local label = 0
        def __init__(self, g):
            super().__init__()
            self.g = g

        def forward(self, z, y):
            return self.g(z, torch.zeros_like(y))

    return ({str(L): _OneClass(g) for L, g in generators.items()},
            {str(L): {0: int(L)} for L in generators})


def unflatten_pair(flat, template_state):
    """Flat vector -> pair state with the template's keys/shapes/dtypes."""
    return unflatten(flat, flatten(template_state)[1])
