"""Class-conditional BatchNorm generator: one shared trunk + a small row of parameters per label.

  trunk  every convolution and the BatchNorm running statistics: shared, aggregated by everyone
  row k  label k's own parameters: an input embedding e_k (emb_dim) and, for every BatchNorm
         layer, a scale gamma_k and shift beta_k (BigGAN / cGAN-style conditional BN)
         -> emb_dim + 2 x (256 + 128 + 64) = 912 numbers for the 32x32 DCGAN
A client's generator holds ONLY the rows of its own labels (local rows 0..k-1). A label's row gets
gradients only from samples of that label, so "only update what belongs to one label" holds for
the rows; the trunk is shared (it learns from every label). The Aggregator's generator holds all
U rows. Rows travel to clients by KEM (secure_cbn), so a client without label D never gets row D.
"""
import numpy as np
import torch
from torch import nn

from .settle import flatten, unflatten

ROW_KEYS = ('emb', 'gamma', 'beta')


class CondBN(nn.Module):
    """BatchNorm without its own affine; per-label gamma/beta rows instead."""

    def __init__(self, features, num_labels, dim=2):
        super().__init__()
        self.bn = (nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d)(features, affine=False)
        self.gamma = nn.Embedding(num_labels, features)
        self.beta = nn.Embedding(num_labels, features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.dim = dim

    def forward(self, x, y):
        shape = (-1, x.size(1)) + (1,) * (x.dim() - 2)
        return self.bn(x) * self.gamma(y).view(shape) + self.beta(y).view(shape)


class CBNGenerator(nn.Module):
    """32x32 DCGAN generator (same widths as nets.DCGANGenerator) with conditional BN."""

    def __init__(self, num_labels, noise_dim=128, emb_dim=16, channels=3, widths=(256, 128, 64)):
        super().__init__()
        self.noise_dim, self.emb_dim = noise_dim, emb_dim
        self.emb = nn.Embedding(num_labels, emb_dim)
        c_in = noise_dim + emb_dim
        self.ups = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i, w in enumerate(widths):
            self.ups.append(nn.ConvTranspose2d(c_in, w, 4, 1 if i == 0 else 2, 0 if i == 0 else 1, bias=False))
            self.norms.append(CondBN(w, num_labels))
            c_in = w
        self.out = nn.ConvTranspose2d(c_in, channels, 4, 2, 1, bias=False)

    def forward(self, z, y):
        x = torch.cat([z.view(z.size(0), -1), self.emb(y)], 1).view(z.size(0), -1, 1, 1)
        for up, norm in zip(self.ups, self.norms):
            x = torch.relu(norm(up(x), y))
        return torch.tanh(self.out(x))


class TinyCBNGenerator(nn.Module):
    """Smoke-test size: 2x2 RGB images."""

    def __init__(self, num_labels, noise_dim=4, emb_dim=2):
        super().__init__()
        self.noise_dim, self.emb_dim = noise_dim, emb_dim
        self.emb = nn.Embedding(num_labels, emb_dim)
        self.hidden = nn.Linear(noise_dim + emb_dim, 8)
        self.norms = nn.ModuleList([CondBN(8, num_labels, dim=1)])
        self.out = nn.Linear(8, 12)

    def forward(self, z, y):
        h = self.norms[0](self.hidden(torch.cat([z, self.emb(y)], 1)), y)
        return torch.tanh(self.out(torch.relu(h))).reshape(-1, 3, 2, 2)


def _is_row(name):
    return name.split('.')[-2] in ('emb', 'gamma', 'beta') and name.endswith('weight')


def trunk_state(g) -> dict:
    """Shared part. BatchNorm num_batches_tracked is left out: unused (momentum is set), and an
    integer counter would only waste upload and break the per-tensor quantization scale."""
    return {k: v.detach().cpu().numpy().copy() for k, v in g.state_dict().items()
            if not _is_row(k) and not k.endswith('num_batches_tracked')}


def rows(g) -> np.ndarray:
    """(num_labels x P) matrix: row k = label k's parameters, fixed order over all layers."""
    sd = g.state_dict()
    return np.concatenate([sd[k].detach().cpu().numpy() for k in sorted(sd) if _is_row(k)], 1).astype(np.float64)


def set_rows(g, table, which=None):
    """Write rows (table[i] into label which[i], default all) back into the generator."""
    sd = g.state_dict()
    which = range(len(table)) if which is None else which
    off = 0
    for k in sorted(sd):
        if _is_row(k):
            w = sd[k].shape[1]
            sd[k][list(which)] = torch.as_tensor(np.asarray(table)[:, off:off + w], dtype=sd[k].dtype)
            off += w
    g.load_state_dict(sd)


def load_trunk(g, state):
    sd = g.state_dict()
    sd.update({k: torch.as_tensor(v) for k, v in state.items()})
    g.load_state_dict(sd)


class ClientCBNGAN:
    """One client: its own generator (shared trunk + its labels' rows) and a private local D."""

    def __init__(self, generator, discriminator, config=None, device='cpu', seed=None):
        config = config or {}
        self.device, self.G, self.D = device, generator.to(device), discriminator.to(device)
        self.noise_dim = config.get('gen_noise_dim', 128)
        self.epochs = config.get('gen_local_epochs', 1)
        args = dict(lr=config.get('gen_lr', 2e-4), betas=(config.get('gan_beta1', .5), config.get('gan_beta2', .999)))
        self.g_opt = torch.optim.Adam(self.G.parameters(), **args)
        self.d_opt = torch.optim.Adam(self.D.parameters(), **args)
        self._ref = None
        self._rng_device = str(device) if str(device).startswith('cuda') else 'cpu'    # keep cuda:k
        self.rng = torch.Generator(device=self._rng_device)
        if seed is not None:
            self.rng.manual_seed(seed)

    def _noise(self, n):
        return torch.randn(n, self.noise_dim, device=self._rng_device, generator=self.rng).to(self.device)

    def load_global(self, trunk: dict, own_rows: np.ndarray):
        """trunk: global trunk state; own_rows: this client's rows (local label order)."""
        load_trunk(self.G, trunk)
        set_rows(self.G, own_rows)
        self._ref = (flatten(trunk_state(self.G))[0], rows(self.G))

    def train(self, loader) -> dict:
        """Returns {local label: samples seen} (first epoch)."""
        bce, counts = nn.BCEWithLogitsLoss(), {}
        self.G.train(); self.D.train()
        for epoch in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if len(x) < 2:
                    continue
                if epoch == 0:
                    for a, c in zip(*np.unique(y.cpu().numpy(), return_counts=True)):
                        counts[int(a)] = counts.get(int(a), 0) + int(c)
                real, fake = torch.ones(len(x), 1, device=self.device), torch.zeros(len(x), 1, device=self.device)
                self.d_opt.zero_grad()
                d_loss = .5 * (bce(self.D(x, y).view(-1, 1), real)
                               + bce(self.D(self.G(self._noise(len(x)), y).detach(), y).view(-1, 1), fake))
                d_loss.backward(); self.d_opt.step()
                self.g_opt.zero_grad()
                bce(self.D(self.G(self._noise(len(x)), y), y).view(-1, 1), real).backward()
                self.g_opt.step()
        return counts

    def update(self):
        """-(local - global) for the trunk and for every own row."""
        t, r = flatten(trunk_state(self.G))[0], rows(self.G)
        return -(t - self._ref[0]), -(r - self._ref[1])

    def state_dict(self) -> dict:
        return dict(G=self.G.state_dict(), D=self.D.state_dict(), g_opt=self.g_opt.state_dict(),
                    d_opt=self.d_opt.state_dict(), rng=self.rng.get_state(),
                    ref=None if self._ref is None else tuple(torch.from_numpy(a) for a in self._ref))

    def load_state_dict(self, state: dict):
        self.G.load_state_dict(state['G']); self.D.load_state_dict(state['D'])
        self.g_opt.load_state_dict(state['g_opt']); self.d_opt.load_state_dict(state['d_opt'])
        self._ref = None if state['ref'] is None else tuple(a.numpy().copy() for a in state['ref'])
        self.rng.set_state(state['rng'])
