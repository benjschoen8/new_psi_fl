"""Phase 3c: fixed-length upload vector in Z_{2^k}.

Layout: [n_{i,L} * clip(grad_L)]_{L} || [n_{i,L}]_{L} || [1_{i,L}]_{L}
Labels a client does not hold are zero, so every upload has the same length.
Gradients use signed fixed point (frac_bits); counts and indicators are plain integers.
The constructor refuses parameters under which the aggregate could wrap around.
"""
import numpy as np


class UploadLayout:
    def __init__(self, params, dims: dict, n_max: int, num_clients: int):
        """params: bb.PublicParams; dims {L: flattened generator size}; n_max: public cap on n_{i,L}."""
        self.labels, self.p = tuple(params.labels), params
        if set(dims) != set(self.labels) or any(d < 1 for d in dims.values()):
            raise ValueError('need a positive dimension for every label')
        if n_max < 1 or num_clients < 1:
            raise ValueError('n_max and num_clients must be positive')
        self.dims, self.n_max = dict(dims), n_max
        self.k, self.f = params.modulus_bits, params.frac_bits
        self.mask = np.uint64((1 << self.k) - 1) if self.k < 64 else np.uint64(2 ** 64 - 1)
        # |sum_i n_i * g_i| per coordinate <= num_clients * n_max * clip; must fit the signed range
        worst = num_clients * n_max * params.clip * 2 ** self.f
        if worst >= 2 ** (self.k - 1) or num_clients * n_max >= 2 ** (self.k - 1):
            raise ValueError(f'overflow: need num_clients*n_max*clip*2^frac < 2^{self.k - 1} '
                             f'(got {worst:.3g}); lower frac_bits/n_max/clip or raise modulus_bits')
        self.offsets, off = {}, 0
        for L in self.labels:
            self.offsets[L] = off
            off += self.dims[L]
        self.grad_size = off
        self.size = off + 2 * len(self.labels)

    def _q(self, x):
        v = np.rint(np.asarray(x, np.float64) * 2 ** self.f).astype(np.int64)
        return v.astype(np.uint64) & self.mask            # two's complement mod 2^k

    def _signed(self, v):
        v = np.asarray(v, np.uint64) & self.mask
        half = np.uint64(1 << (self.k - 1))
        out = v.astype(np.int64) if self.k < 64 else v.view(np.int64)
        if self.k < 64:
            out = np.where(v >= half, out - (1 << self.k), out)
        return out

    def clip(self, g):
        g = np.asarray(g, np.float64).ravel()
        norm = np.linalg.norm(g)
        return g * min(1.0, self.p.clip / norm) if norm > 0 else g

    def encode(self, grads: dict, counts: dict) -> np.ndarray:
        """grads {L: gradient of theta_L}, counts {L: n_{i,L}} for held labels only."""
        if set(grads) != set(counts) or not set(grads) <= set(self.labels):
            raise ValueError('grads and counts must cover the same known labels')
        vec = np.zeros(self.size, np.uint64)
        n_off = self.grad_size
        for L, g in grads.items():
            n = int(counts[L])
            if not 1 <= n <= self.n_max:
                raise ValueError(f'{L}: count must be in [1, n_max]')
            g = self.clip(g)
            if g.size != self.dims[L]:
                raise ValueError(f'{L}: gradient size {g.size} != {self.dims[L]}')
            o = self.offsets[L]
            vec[o:o + g.size] = self._q(n * g)
            k = self.labels.index(L)
            vec[n_off + k] = n
            vec[n_off + len(self.labels) + k] = 1
        return vec

    def decode(self, total) -> tuple:
        """Aggregate vector -> ({L: sum_i n g}, {L: N_L}, {L: |Group_L|})."""
        total = np.asarray(total, np.uint64)
        if total.shape != (self.size,):
            raise ValueError('aggregate length mismatch')
        signed = self._signed(total)
        grads = {L: signed[o:o + self.dims[L]] / 2 ** self.f for L, o in self.offsets.items()}
        n_off, m = self.grad_size, len(self.labels)
        N = {L: int(signed[n_off + k]) for k, L in enumerate(self.labels)}
        G = {L: int(signed[n_off + m + k]) for k, L in enumerate(self.labels)}
        return grads, N, G

    def add(self, a, b):
        return (np.asarray(a, np.uint64) + np.asarray(b, np.uint64)) & self.mask
