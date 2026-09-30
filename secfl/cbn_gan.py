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
import threading
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch import nn

from .settle import flatten, unflatten

ROW_KEYS = ('emb', 'gamma', 'beta')


class _RWLock:
    """Client threads share the GPU; a CUDA-graph capture must not see other threads' CUDA calls.
    Training steps take the shared side, a capture the exclusive side (once per client)."""

    def __init__(self):
        self._c, self._readers, self._writer, self._waiting = threading.Condition(), 0, False, 0

    @contextmanager
    def read(self):
        with self._c:
            while self._writer or self._waiting:
                self._c.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._c:
                self._readers -= 1
                self._c.notify_all()

    @contextmanager
    def write(self):
        with self._c:
            self._waiting += 1
            while self._writer or self._readers:
                self._c.wait()
            self._waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._c:
                self._writer = False
                self._c.notify_all()


_GPU = _RWLock()
GRAPH_WARMUP = 3                                  # eager steps (side stream) before a capture


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


class PerLabelGenerator(nn.Module):
    """One whole generator per label (no shared weights): row k = every parameter of label k's own
    copy of `template` (flattened into emb.weight[k], so rows()/set_rows()/KEM/SecAgg treat it as a
    CBN row). All rows start from the same template init, so a client's local rows equal the public
    initial rows of any union index. BatchNorm uses batch statistics (no running stats to share).
    The trunk is one unused buffer: nothing is shared between labels."""

    def __init__(self, num_labels, template):
        super().__init__()
        self.noise_dim = template.noise_dim
        self._t = [template]                                   # not a submodule: its weights live in the rows
        self._shapes = [(n, p.shape) for n, p in template.named_parameters()]
        flat = torch.cat([p.detach().reshape(-1) for p in template.parameters()])
        self.emb = nn.Embedding(num_labels, flat.numel())
        with torch.no_grad():
            self.emb.weight.copy_(flat.expand(num_labels, -1))
        self.trunk = nn.Module()
        self.trunk.register_buffer('unused', torch.zeros(1))  # ponytail: keeps the trunk path non-empty

    def extra_repr(self):
        return f'template={self._t[0]}'

    def forward(self, z, y):
        if isinstance(self._t[0], DCGANTemplate) and (z.is_cuda or len(y.unique()) > 4):   # CPU, few labels:
            return self._dcgan(z, y)                                                     # the loop is cheaper
        from torch.func import functional_call
        out = None
        for k in y.unique():                                   # ponytail: one pass per label in the batch
            m = y == k
            parts = self.emb.weight[k].split([s.numel() for _, s in self._shapes])
            o = functional_call(self._t[0], {n: p.view(s) for (n, s), p in zip(self._shapes, parts)}, (z[m],))
            if out is None:
                out = o.new_empty((len(z),) + o.shape[1:])
            out[m] = o
        return out

    def _dcgan(self, z, y):
        """Same function as the loop, whole batch at once: every sample's own weights (its label's row),
        grouped transposed convolutions (groups = batch), BatchNorm statistics per label group."""
        F = nn.functional
        B, inv, L = len(z), y, self.emb.num_embeddings          # static shapes: CUDA-graph capturable
        cnt = torch.zeros(L, 1, dtype=z.dtype, device=z.device).index_add_(
            0, y, torch.ones(B, 1, dtype=z.dtype, device=z.device)).clamp_min_(1)
        parts = dict(zip([n for n, _ in self._shapes],
                         self.emb.weight[y].split([s.numel() for _, s in self._shapes], 1)))
        x, mods = z.view(B, -1, 1, 1), list(self._t[0].net)
        for i, m in enumerate(mods):
            if isinstance(m, nn.ConvTranspose2d):
                cin, cout = m.in_channels, m.out_channels
                w = parts[f'net.{i}.weight'].reshape(B * cin, cout, *m.kernel_size)
                x = F.conv_transpose2d(x.reshape(1, B * cin, *x.shape[2:]), w, stride=m.stride,
                                       padding=m.padding, groups=B)
                x = x.view(B, cout, *x.shape[2:])
            elif isinstance(m, nn.BatchNorm2d):                   # batch stats over each label's samples
                s1 = torch.zeros(len(cnt), x.size(1), dtype=x.dtype, device=x.device).index_add_(0, inv, x.mean((2, 3)))
                s2 = torch.zeros_like(s1).index_add_(0, inv, (x * x).mean((2, 3)))
                mean = (s1 / cnt)[inv]
                var = ((s2 / cnt)[inv] - mean * mean).clamp_min(0)
                x = ((x - mean[..., None, None]) * torch.rsqrt(var + m.eps)[..., None, None]
                     * parts[f'net.{i}.weight'][..., None, None] + parts[f'net.{i}.bias'][..., None, None])
            else:
                x = m(x)
        return x


class DCGANTemplate(nn.Module):
    """32x32 DCGAN generator for PerLabelGenerator (unconditional, batch-stat BatchNorm)."""

    def __init__(self, noise_dim=128, channels=3, widths=(64, 32, 16)):
        super().__init__()
        self.noise_dim, layers, c = noise_dim, [], noise_dim
        for i, w in enumerate(widths):
            layers += [nn.ConvTranspose2d(c, w, 4, 1 if i == 0 else 2, 0 if i == 0 else 1, bias=False),
                       nn.BatchNorm2d(w, track_running_stats=False), nn.ReLU()]
            c = w
        self.net = nn.Sequential(*layers, nn.ConvTranspose2d(c, channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class TinyTemplate(nn.Module):
    """Smoke-test size: 2x2 RGB images."""

    def __init__(self, noise_dim=4):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(nn.Linear(noise_dim, 8), nn.ReLU(), nn.Linear(8, 12), nn.Tanh())

    def forward(self, z):
        return self.net(z).reshape(-1, 3, 2, 2)


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
            sd[k][list(which)] = torch.as_tensor(np.asarray(table)[:, off:off + w], dtype=sd[k].dtype, device=sd[k].device)
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
        self.config = config
        self.device, self.G, self.D = device, generator.to(device), discriminator.to(device)
        self.noise_dim = config.get('gen_noise_dim', 128)
        self.epochs = config.get('gen_local_epochs', 1)
        cuda = str(device).startswith('cuda')
        # CUDA speed-ups (same maths): fused Adam = one kernel per optimizer step; a CUDA graph replays a
        # whole D+G training step with one launch instead of ~1,100 (the step was launch-bound)
        self.graphs = cuda and config.get('cuda_graph', True)
        args = dict(lr=config.get('gen_lr', 2e-4), betas=(config.get('gan_beta1', .5), config.get('gan_beta2', .999)))
        if cuda and config.get('fused_adam', True):
            args.update(fused=True, capturable=self.graphs)
        elif self.graphs:
            args.update(capturable=True)
        self.g_opt = torch.optim.Adam(self.G.parameters(), **args)
        self.d_opt = torch.optim.Adam(self.D.parameters(), **args)
        self._graph, self._static, self._warm = None, None, 0
        self._ref = None
        self.guide, self.guide_weight = None, 0.   # heter: frozen local classifier guiding G (set_guide)
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

    def rebase(self, trunk_flat: np.ndarray, own_rows: np.ndarray):
        """Keep the (pretrained) weights; the next update() is measured from this global state
        (used after the union: own_rows = the public initial rows of this client's union indices)."""
        self._ref = (np.asarray(trunk_flat, np.float64).copy(), np.asarray(own_rows, np.float64).copy())

    def train(self, loader) -> dict:
        """Returns {local label: samples seen} (first epoch)."""
        counts = {}
        self.G.train(); self.D.train()
        full = getattr(loader, 'batch_size', None)
        for epoch in range(self.epochs):
            for x, y in loader:
                if len(x) < 2:
                    continue
                if epoch == 0:                                   # counted on the CPU copy: no GPU sync
                    for a, c in zip(*np.unique((y.cpu() if y.is_cuda else y).numpy(), return_counts=True)):
                        counts[int(a)] = counts.get(int(a), 0) + int(c)
                if self.graphs and len(x) == full:
                    self._graph_step(x, y)
                else:
                    with _GPU.read() if self.graphs else nullcontext():
                        z1, z2 = self._noise(len(x)), self._noise(len(x))   # same draws and order as before
                        self._step(x.to(self.device), y.to(self.device), z1, z2)
        return counts

    def _step(self, x, y, z1, z2):
        """One D step then one G step (the whole update; also what a CUDA graph records)."""
        bce = nn.functional.binary_cross_entropy_with_logits
        real, fake = torch.ones(len(x), 1, device=x.device), torch.zeros(len(x), 1, device=x.device)
        self.d_opt.zero_grad()
        d_loss = .5 * (bce(self.D(x, y).view(-1, 1), real)
                       + bce(self.D(self.G(z1, y).detach(), y).view(-1, 1), fake))
        d_loss.backward(); self.d_opt.step()
        self.g_opt.zero_grad()
        fake_x = self.G(z2, y)
        g_loss = bce(self.D(fake_x, y).view(-1, 1), real)
        if self.guide is not None:                        # heter: the client's own classifier must
            out = self.guide(fake_x)                      # recognise the generated class
            g_loss = g_loss + self.guide_weight * nn.functional.cross_entropy(
                out[1] if isinstance(out, tuple) else out, y)
        g_loss.backward()
        self.g_opt.step()

    def _graph_step(self, x, y):
        """Full batches: GRAPH_WARMUP ordinary steps on a side stream, then the step is captured once
        and replayed (inputs copied into fixed buffers). Any capture failure -> ordinary steps.
        Every GPU call sits under the lock (shared for steps, exclusive for the capture)."""
        dev = torch.device(self.device)
        n = len(x)
        if self._graph is not None or self._warm < GRAPH_WARMUP:
            with _GPU.read(), torch.cuda.device(dev):
                z1, z2 = self._noise(n), self._noise(n)
                if self._graph is not None:
                    for buf, v in zip(self._static, (x, y, z1, z2)):
                        buf.copy_(v)
                    self._graph.replay()
                    return
                side = torch.cuda.Stream(dev)             # warm-up: real steps, on a side stream
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):
                    self._step(x.to(dev), y.to(dev), z1, z2)
                torch.cuda.current_stream(dev).wait_stream(side)
                self._warm += 1
                return
        with _GPU.write(), torch.cuda.device(dev):        # capture once; other clients wait
            z1, z2 = self._noise(n), self._noise(n)
            self._static = [v.to(dev).clone() for v in (x, y, z1, z2)]
            self.d_opt.zero_grad(set_to_none=True); self.g_opt.zero_grad(set_to_none=True)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                    self._step(*self._static)
            except Exception as e:                        # recorded, not run: weights untouched
                print(f'[cuda graph] capture failed ({type(e).__name__}: {e}); ordinary steps from now on',
                      flush=True)
                self.graphs, self._static = False, None
                self._step(x.to(dev), y.to(dev), z1, z2)
                return
            self._graph = graph
            graph.replay()                                # this batch's step

    def _drop_graph(self):
        """Optimizer state or guide replaced: the recorded graph points at old tensors."""
        self._graph, self._static, self._warm = None, None, 0

    def set_guide(self, classifier, weight):
        """Frozen local classifier (never uploaded): G's loss += weight * CE(classifier(G(z, y)), y)."""
        self._drop_graph()
        classifier = classifier.to(self.device).eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        self.guide, self.guide_weight = classifier, float(weight)

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
        self._drop_graph()
        self._ref = None if state['ref'] is None else tuple(a.numpy().copy() for a in state['ref'])
        self.rng.set_state(state['rng'])
