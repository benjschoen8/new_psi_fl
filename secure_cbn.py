"""No-cluster protocol with a class-conditional-BatchNorm generator (secfl.cbn_gan):
one shared trunk + one small parameter row per label, rows delivered by KEM.

Setup (clients only):
  1. label union by OPRF (label_union.oprf_union_with_keys, optional weak image check): every
     client gets the index k of each of its labels and sk_k = KDF(tag); the Aggregator gets U and
     pk_0..pk_{U-1} (through SecAgg with uniform masks: no sender, no holder count), nothing else
  2. public init: trunk from a public seed, every row = (emb from the seed, gamma = 1, beta = 0)
Every round:
  3. downlink: trunk broadcast in the clear; row k KEM-sealed under pk_k on the bulletin board,
     so only holders of label k can open it (a client with A, B, C never sees row D)
  4. clients train their generator (trunk + own rows) against a private local D; a client only
     holds, trains and uploads the rows of its own labels (within a client, rows interact through
     the shared BatchNorm statistics)
  5. upload through SecAgg, one fixed-length vector from EVERY client (zeros where it has nothing):
     trunk update + all U rows (own rows filled) + U+1 contributor counts. The Aggregator takes the
     mean over contributors and updates a row only if >= min_holders clients contributed to it.
       quantize=True (default): a public per-round random keep_frac of the trunk and of the row
         coordinates; clipped 8-bit stochastic rounding with public per-tensor scales; 16-bit SecAgg
       quantize=False: every coordinate, 64-bit fixed point (2^-24)
     Same contributor mean and same participants in both, so the two differ only by compression.
  6. Aggregator trains the global classifier on G(z, k) for every index k (it holds all rows)
  7. evaluation (evaluation.evaluate_global); union quality (label_union.index_metrics) is
     computed once, by the experimenter

agg='plain' is Plain-GeFL, the no-cryptography baseline with the same model and the same update
rule: plaintext union of (name, image code), rows handed out directly, plain contributor mean.

Leakage to the Aggregator (secagg): U, the pks, the global model, and per round the number of
clients that contributed to each row (never who). Clients learn U.

Outputs are split by role:  checkpoint_last.pt (Aggregator state), clients/round_XXXX.pt (each
client's private state: keys, index, G, D, optimisers), evaluator/ (ownership tables, union
metrics: experimenter only). This is a single-process simulation of all parties.

  python -m secure_code_no_cluster --gen cbn --smoke --rounds 3        # CLI lives there
"""
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from evaluation import evaluate_global
from label_union import index_metrics
from label_union.oprf_union import oprf_union_with_keys, canonical, pairs_union_with_keys, plain_pairs_grouping
from secfl import kem, compress
from secfl.bb import BulletinBoard
from secfl.cbn_gan import ClientCBNGAN, trunk_state, rows, set_rows, load_trunk, _is_row
from secfl.secagg import run_secagg
from secfl.settle import flatten, unflatten
from secure_code_no_cluster import (seeded, _rng_state, _set_rng_state, union_relations, write_json, label_samples,
                                    format_union)
from training import GlobalClassifierTrainer

CHECKPOINT_FORMAT = 'secure_cbn.v2'
FRAC_BITS, CLIP = 24, 1e3                                                 # 64-bit fixed point
MAX_CLIENTS_16BIT = (1 << 15) // compress.LEVELS                          # 258: no overflow of 127 n


# ---------------------------------------------------------------------------- files
def _save(path, obj):
    """Atomic and durable: temp file, fsync, rename, fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load(path, fmt=CHECKPOINT_FORMAT):
    state = torch.load(path, map_location='cpu', weights_only=False)      # own files only
    if state.get('format') != fmt:
        raise ValueError(f'{path}: not a {fmt} file')
    return state


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------- upload encoding
def row_spec(g):
    """(name, shape, dtype) of every block of a row, in rows() order (for per-tensor scales). A per-label
    generator's row is its whole generator: one block per parameter tensor of the template (emb.weight
    holds them in template order), not one block (one scale) for the whole generator."""
    from secfl.cbn_gan import PerLabelGenerator
    if isinstance(g, PerLabelGenerator):
        return [(n, (int(np.prod(sh)),), np.float32) for n, sh in g._shapes]
    sd = g.state_dict()
    return [(k, (sd[k].shape[1],), np.float32) for k in sorted(sd) if _is_row(k)]


def _kept(g, idx):
    """The uploaded coordinates idx of an update: clients send only these (already sliced, same length as
    idx) since keep indices are public before training; a full-length update (comm-only) is sliced here."""
    return g if g.size == idx.size else g[idx]


def _memory(devices):
    """This process's peak / current host memory and every CUDA device's peak and reserved memory (MB)."""
    import resource
    out = dict(host_peak_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)   # Linux: KB
    try:
        with open('/proc/self/status') as f:
            out['host_MB'] = next(int(l.split()[1]) for l in f if l.startswith('VmRSS')) / 1024
    except (OSError, StopIteration):
        pass
    for d in sorted({str(d) for d in devices if str(d).startswith('cuda')}):
        out[d] = dict(peak_MB=torch.cuda.max_memory_allocated(d) / 2**20,
                      reserved_MB=torch.cuda.memory_reserved(d) / 2**20)
    return out


def encode_update(grads, U, it, ir, sc_t, sc_r, rng, fixed=False, blocks=None, res=compress.LEVELS):
    """grads {'T': trunk update, k: row-k update} (missing = zeros) -> uint64 vector:
    [trunk[it] | row 0[ir] | ... | row U-1[ir] | counts (trunk, rows) | clip fractions].
    fixed=False: clipped 8-bit stochastic rounding, mod 2^16. fixed=True: 2^-24 fixed point, mod 2^64.
    blocks (tb, rb, nt, nr): tensor-block id of every uploaded trunk / row coordinate; then the client
    also reports, per block, the fraction of its values that exceeded the scale (in steps of 1/res, adaptive
    clipping feedback: the Aggregator sees only the sum over clients; res * clients < 2^15)."""
    kt, kr = it.size, ir.size
    extra = 0 if blocks is None or fixed else blocks[2] + blocks[3]
    base = kt + U * kr + U + 1
    v = np.zeros(base + extra, np.int64)
    q = ((lambda g, c: np.round(np.clip(g, -CLIP, CLIP) * (1 << FRAC_BITS)).astype(np.int64)) if fixed
         else (lambda g, c: compress.quantize(g, c, rng)))
    if extra:
        tb, rb, nt, nr = blocks
        rc, rn = np.zeros(nr), np.zeros(nr)
    for k, g in grads.items():
        if k == 'T':
            g = _kept(g, it)
            v[:kt] = q(g, sc_t[it])
            v[kt + U * kr] = 1
            if extra:
                over = np.abs(g) > sc_t[it]
                tot = np.bincount(tb, minlength=nt)
                v[base:base + nt] = np.round(np.bincount(tb[over], minlength=nt) / np.maximum(tot, 1) * res)
        else:
            g = _kept(g, ir)
            v[kt + k * kr:kt + (k + 1) * kr] = q(g, sc_r[k][ir])
            v[kt + U * kr + 1 + k] = 1
            if extra:
                rc += np.bincount(rb[np.abs(g) > sc_r[k][ir]], minlength=nr)
                rn += np.bincount(rb, minlength=nr)
    if extra:
        v[base + nt:] = np.round(rc / np.maximum(rn, 1) * res)
    return v.view(np.uint64).copy() if fixed else (v % (1 << compress.BITS)).astype(np.uint64)


def decode_update(total, U, it, ir, sc_t, sc_r, fixed=False, blocks=None, res=compress.LEVELS):
    """SecAgg sum -> (mean trunk update on it or None, {k: mean row-k update on ir}, counts[U+1]);
    with blocks also (mean clip fraction per trunk block, per row block) over the clients that trained."""
    kt, kr = it.size, ir.size
    if fixed:
        v = np.asarray(total, np.uint64).view(np.int64).astype(np.float64) / (1 << FRAC_BITS)
        n = np.asarray(total, np.uint64)[kt + U * kr:].astype(np.int64)
        t_part, r_part = v[:kt], [v[kt + k * kr:kt + (k + 1) * kr] for k in range(U)]
    else:
        w = np.asarray(total, np.int64) % (1 << compress.BITS)
        w = np.where(w >= 1 << (compress.BITS - 1), w - (1 << compress.BITS), w)
        n = w[kt + U * kr:]
        t_part = w[:kt] / compress.LEVELS * sc_t[it]
        r_part = [w[kt + k * kr:kt + (k + 1) * kr] / compress.LEVELS * sc_r[k][ir] for k in range(U)]
    n, clip = n[:U + 1], n[U + 1:]
    t = t_part / n[0] if n[0] else None
    r = {k: r_part[k] / n[k + 1] for k in range(U) if n[k + 1]}
    if blocks is None or fixed:
        return t, r, n
    frac = clip / res / max(n[0], 1)
    return t, r, n, (frac[:blocks[2]], frac[blocks[2]:])


def plain_mean(results, U, it, ir):
    """Plain-GeFL: the same contributor mean, in the clear."""
    n, st, sr = np.zeros(U + 1, np.int64), np.zeros(it.size), np.zeros((U, ir.size))
    for _, u in results:
        for k, g in (u[0] if u else {}).items():
            if k == 'T':
                st += _kept(g, it); n[0] += 1
            else:
                sr[k] += _kept(g, ir); n[k + 1] += 1
    return (st / n[0] if n[0] else None), {k: sr[k] / n[k + 1] for k in range(U) if n[k + 1]}, n


# ---------------------------------------------------------------------------- union
def _view(names, index, ids, U):
    return [dict(cls=k, label='/'.join(sorted({x for l, i in zip(names, index) for x in l if i[x] == k})),
                 holders=sum(k in i.values() for i in index), clients=[c for c, i in zip(ids, index) if k in i.values()])
            for k in range(U)]


def _domains(samples, names):
    if samples is None:
        return None
    from label_union.domain import client_codes
    return [client_codes(s, l)[0] for s, l in zip(samples, names)]


def _image_sets(samples, names, image):
    """image = (k, t): per client {label: its k nearest public image anchors} (t-out-of-k matching),
    or None (no images, or image=None: the one-code check of _domains)."""
    if samples is None or not image:
        return None
    from label_union.domain import client_sets
    return [client_sets(s, l, image[0]) for s, l in zip(samples, names)]


def _pairs_result(names, dictionary, ids, keys, sets, t, secure, workers, method, say, t0):
    """Union with t-out-of-k image matching (label_union.oprf_union.pairs_union_with_keys); secure=False:
    the same grouping in the clear (Plain-GeFL)."""
    if secure:
        index, sks, pks, info = pairs_union_with_keys(keys, sets, t, workers=workers)
        U = len(pks)
    else:
        (index, U), sks, pks = plain_pairs_grouping(keys, sets, t), None, None
        info = dict(setup_upload_bytes_per_client=0, setup_download_bytes_per_client=0)
    info = dict(info, method=method, image_match=f'{t}-of-{len(next(iter(sets[0].values())))}',
                seconds=time.perf_counter() - t0)
    metrics = index_metrics(names, index, U, dictionary)
    say(f"[setup] {method} union done in {info['seconds']:.1f}s: {U} labels; exact={metrics['exact']}")
    return dict(index=index, sks=sks, pks=pks, U=U, metrics=metrics, view=_view(names, index, ids, U), info=info)


def cbn_union(names, dictionary, ids, workers=1, samples=None, say=lambda *_: None, image=None):
    """OPRF union with KEM keys. Returns a dict stored in checkpoints (never recomputed on resume)."""
    sets = _image_sets(samples, names, image)
    if sets:
        say(f'[setup] label union: {len(ids)}-party OPRF tags + SecAgg, image match {image[1]}-of-{image[0]} ...')
        return _pairs_result(names, dictionary, ids, [{x: x for x in l} for l in names], sets, image[1], True,
                             workers, 'oprf-pairs', say, time.perf_counter())
    domains = _domains(samples, names)
    say(f'[setup] label union: {len(ids)}-party OPRF tags + SecAgg{" + image check" if domains else ""} ...')
    t = time.perf_counter()
    index, sks, pks, info = oprf_union_with_keys(names, workers=workers, domains=domains)
    U = len(pks)
    metrics = index_metrics(names, index, U, dictionary)
    info = dict(info, method='oprf', seconds=time.perf_counter() - t)
    say(f"[setup] label union done in {info['seconds']:.1f}s: Aggregator table has {U} indices; "
        f"exact={metrics['exact']}")
    return dict(index=index, sks=sks, pks=pks, U=U, metrics=metrics, view=_view(names, index, ids, U), info=info)


def plain_union(names, dictionary, ids, samples=None, say=lambda *_: None, image=None):
    """Plain-GeFL: union of (canonical name, image code) in the clear, no keys."""
    sets = _image_sets(samples, names, image)
    if sets:
        return _pairs_result(names, dictionary, ids, [{x: canonical(x).decode() for x in l} for l in names], sets,
                             image[1], False, 0, 'plain-pairs', say, time.perf_counter())
    domains = _domains(samples, names)
    t = time.perf_counter()
    key = lambda i, x: canonical(x) + (b'\x00' + domains[i][x].encode() if domains else b'')
    order = {v: k for k, v in enumerate(sorted({key(i, x) for i, l in enumerate(names) for x in l}))}
    index = [{x: order[key(i, x)] for x in l} for i, l in enumerate(names)]
    U = len(order)
    metrics = index_metrics(names, index, U, dictionary)
    say(f"[setup] plaintext union: {U} indices, exact={metrics['exact']}")
    info = dict(method='plain', seconds=time.perf_counter() - t,
                setup_upload_bytes_per_client=0, setup_download_bytes_per_client=0)
    return dict(index=index, sks=None, pks=None, U=U, metrics=metrics, view=_view(names, index, ids, U), info=info)


def circuit_union(names, dictionary, ids, keywords=None, workers=1, samples=None, say=lambda *_: None,
                  image=(6, 2), secure=True, tau=None):
    """Label union by circuit PSI (label_union.circuit_union): grouping inside an honest-majority MPC of the
    clients (exact names, or fuzzy keywords by CSLS distance when keywords are given; plus t-of-k image
    anchors), then the bucket-union and pk SecAggs. secure=False: the same grouping in the clear."""
    from label_union.circuit_union import circuit_union_with_keys, TAU
    sets = _image_sets(samples, names, image)
    fuzzy = keywords is not None
    kw = keywords if fuzzy else [{x: x for x in l} for l in names]
    t0 = time.perf_counter()
    say(f"[setup] label union: circuit PSI ({'fuzzy' if fuzzy else 'exact'} keywords"
        f"{f', image {image[1]}-of-{image[0]}' if sets else ''}) + SecAgg ...")
    index, sks, pks, U, info = circuit_union_with_keys(names, kw, sets, fuzzy, TAU if tau is None else tau,
                                                       image[1] if sets else 2, workers=workers, secure=secure)
    info = dict(info, method=info['method'] + ('-fuzzy' if fuzzy else ''), seconds=time.perf_counter() - t0,
                image_match=f'{image[1]}-of-{image[0]}' if sets else 'off')
    metrics = index_metrics(names, index, U, dictionary)
    say(f"[setup] circuit-PSI union done in {info['seconds']:.1f}s: {U} labels; exact={metrics['exact']}"
        + (f"; MPC (MP-SPDZ, measured) {info['mpc']['measured']['time_seconds']:.1f}s, "
           f"{info['mpc']['measured']['party0_MB']:.0f} MB sent by party 0"
           if info.get('mpc', {}).get('measured') else
           f"; MPC (estimated) {info['mpc']['mults'] / 1e6:.1f}M mults, "
           f"{info['mpc']['bytes_per_client'] / 1e6:.0f} MB/client" if 'mpc' in info else ''))
    return dict(index=index, sks=sks, pks=pks, U=U, metrics=metrics, view=_view(names, index, ids, U), info=info)


def fuzzy_union(names, dictionary, ids, keywords, workers=1, samples=None, say=lambda *_: None, fuzzy=None,
                secure=True, image=None):
    """No dictionary: keywords = per client {label: its own keyword, any language}. Each client snaps
    its keywords to public anchor classes locally (label_union.fuzzy_union), then the exact OPRF
    union runs on the class ids: same leakage as exact. names are only the ground truth for the
    experimenter's metrics. secure=False: the same grouping in the clear (Plain-GeFL)."""
    from label_union import fuzzy_union as fz
    p = dict(fz.params(), **(fuzzy or {}))
    sets = _image_sets(samples, names, image)
    domains = None if sets else _domains(samples, names)
    say(f"[setup] fuzzy union: {p['anchors']} anchors, merge={p['merge']}, floor={p['floor']}"
        f"{f' + image match {image[1]}-of-{image[0]}' if sets else ' + image check' if domains else ''} ...")
    t = time.perf_counter()
    keys = fz.local_keys(names, keywords, p, domains)                    # each client, locally
    if sets:
        return _pairs_result(names, dictionary, ids, keys, sets, image[1], secure, workers,
                             'fuzzy-anchor-pairs' if secure else 'plain-fuzzy-pairs', say, t)
    local = time.perf_counter() - t
    index, sks, pks, U, stats = fz.union(keys, workers=workers, secure=secure)
    info = dict(stats, method='fuzzy-anchor' if secure else 'plain-fuzzy', fuzzy=p, local_seconds=local,
                seconds=time.perf_counter() - t)
    metrics = index_metrics(names, index, U, dictionary)
    say(f"[setup] fuzzy union done in {info['seconds']:.1f}s (local snapping {local:.1f}s): {U} labels; "
        f"exact={metrics['exact']}")
    return dict(index=index, sks=sks, pks=pks, U=U, metrics=metrics, view=_view(names, index, ids, U), info=info)


# ---------------------------------------------------------------------------- run
def data_hash(loader):
    """sha256 of every (image, label) the client trains on, in dataset order, as the model sees them."""
    from torch.utils.data import DataLoader
    h = hashlib.sha256()
    batches = loader.ordered(1024) if hasattr(loader, 'ordered') else \
        DataLoader(loader.dataset, batch_size=1024, shuffle=False, num_workers=0)   # same bytes either way
    for x, y in batches:
        h.update(np.ascontiguousarray(x.numpy(), np.float32).tobytes())
        h.update(np.ascontiguousarray(y.numpy(), np.int64).tobytes())
    return h.hexdigest()


SPEED_ONLY = ('cuda_graph', 'fused_adam')     # config keys that only change speed: not part of the run's
                                              # identity (resume, generator cache keys stay valid)
GENERATOR_CACHE_VERSION = 2          # bump when ClientCBNGAN.train changes: old cache entries stop matching (2: class-balanced batches)


def _finite_state(state):
    """True if every float tensor in a (nested) state dict is finite."""
    if isinstance(state, dict):
        return all(_finite_state(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return all(_finite_state(v) for v in state)
    if torch.is_tensor(state) and state.is_floating_point():
        return bool(torch.isfinite(state).all())
    return True


def train_guide(model, loader, epochs, lr, device, seed):
    """heter: train a client's own classifier on its real local data (own shuffling, so the GAN's data
    order is untouched); returns its training accuracy in the last epoch."""
    from torch.utils.data import DataLoader
    bs = getattr(loader, 'batch_size', None) or 64
    gen = torch.Generator().manual_seed(seed)                            # one stream over all epochs
    if hasattr(loader, 'shuffled'):
        epoch = lambda: loader.shuffled(gen, bs)
    else:
        dl = DataLoader(loader.dataset, batch_size=bs, shuffle=True, generator=gen, num_workers=0)
        epoch = lambda: dl
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hit = n = 0
    for e in range(epochs):
        for x, y in epoch():
            if len(x) < 2:
                continue
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[1] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if e == epochs - 1:
                hit += int((logits.argmax(1) == y).sum()); n += len(y)
    return hit / max(n, 1)


def cached_epochs(cache, key, seed):
    """Epoch counts cached for this client key: generator_<key>_epochs<e>_seed<seed>.pt."""
    out = []
    for f in Path(cache).glob(f'generator_{key}_epochs*_seed{seed}.pt'):
        e = f.name[len(f'generator_{key}_epochs'):-len(f'_seed{seed}.pt')]
        if e.isdigit():
            out.append(int(e))
    return sorted(out)


class _Lock:
    """One process trains a client key at a time (parallel runs of the same clients share the work).
    A lock whose process is gone (same host) or that was not refreshed for `stale` s is taken over."""

    def __init__(self, path, stale=7200):
        self.path, self.stale, self.mine = Path(path), stale, False

    def acquire(self):
        import socket
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f'{socket.gethostname()} {os.getpid()}'.encode())
                os.close(fd)
                self.mine = True
                return True
            except FileExistsError:
                if self._stale():
                    self.path.unlink(missing_ok=True)
                    continue
                return False

    def _stale(self):
        import socket
        try:
            host, pid = self.path.read_text().split()
            age = time.time() - self.path.stat().st_mtime
        except (OSError, ValueError):
            return False
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        return age > self.stale

    def refresh(self):
        if self.mine:
            os.utime(self.path)

    def release(self):
        if self.mine:
            self.path.unlink(missing_ok=True)
            self.mine = False


def run(clients, label_spaces, tests, gen_factory, disc_factory, classifier_factory, config, dictionary,
        agg='secagg', rounds=3, device='cpu', record=None, devices=None, workers=1, checkpoint_dir=None,
        save_every=1, keep_all=False, resume=None, seed=0, progress=False, on_resume=None, domain_check=True,
        samples_per_label=16, union_result=None, quantize=True, keep_frac=0.1, quant_scale0=0.05,
        min_holders=2, warmup_epochs=0, union='exact', keywords=None, fuzzy=None, generator_cache=None,
        cache_every=5, guide_factory=None, guide_epochs=5, guide_weight=.5, client_procs=None, image_match=(6, 2), label_psi='circuit', circuit_tau=None, diagnostics=False,
        comm_only=False):
    """gen_factory(num_rows) -> CBN generator; disc_factory(k) -> local D over k labels.
    agg: 'secagg' (the protocol) or 'plain' (Plain-GeFL, no cryptography).
    union_result: reuse a cbn_union() output (the OPRF indices are random per run).
    quantize / keep_frac / quant_scale0: compressed upload (both: plain sends the same 8-bit values in
    the clear, own rows only; quantize=False: plain floats / secagg 64-bit fixed point).
    min_holders: a row is updated only if at least this many clients contributed this round.
    warmup_epochs: BEFORE the label union every client trains its own generator (public trunk + one row
    per LOCAL label) for this many epochs, nothing sent. After the union each client moves its local row a to union
    row index[a] (a relabel: the condition is a row lookup) and measures round 1's update from the
    public initial state, so round 1 carries the warm-up progress.
    generator_cache: folder of warmed-up client generators, generator_<key>_epochs<e>_seed<seed>.pt,
    key = hash of the client's data (every image and label), its label count, the config, the public
    initial trunk, the model shapes and its random seeds. Each client loads the most epochs cached
    (<= warmup_epochs) and trains only the rest, saving every cache_every epochs; runs of the same
    clients (plain / exact / fuzzy, other warm-up lengths, restarts) share it, and a lock file lets
    parallel runs train each client once.
    guide_factory (heter version): guide_factory(client_id, num_local_labels) -> classifier. Each client
    first trains it on its real local data for guide_epochs (cached like the generators), freezes it,
    and adds guide_weight * CE(classifier(G(z, y)), y) to its generator loss. Stays local, never sent.
    union: 'exact' (names, OPRF) or 'fuzzy' (no dictionary: keywords = per client {label: keyword}
    in the client's own words, snapped to public anchor classes; fuzzy overrides params).
    comm_only: per-round communication benchmark: no training. Every client uploads a random update of
    the real shapes (trunk + one row per own label) and the round runs downlink, encoding, SecAgg (or the
    plain sum) and decoding exactly as in training; no global classifier, no evaluation (accuracy None)."""
    if agg not in ('plain', 'secagg'):
        raise ValueError('agg must be plain or secagg')
    quant = quantize                                                      # plain too: same compression
    if quant and len(clients) > MAX_CLIENTS_16BIT:
        raise ValueError(f'16-bit SecAgg holds sums of at most {MAX_CLIENTS_16BIT} clients; use quantize=False')
    devices = list(devices or [device])
    ids = [c.id for c in clients]
    names = [list(label_spaces[i]) for i in ids]
    from tqdm.auto import tqdm
    say = tqdm.write if progress else (lambda *_: None)
    manifest = dict(ids=ids, names=_digest(names), data=[len(c.train_loader.dataset) for c in clients],
                    agg=agg, quantize=quant, keep_frac=keep_frac if quant else 1.0, quant_scale0=quant_scale0,
                    min_holders=min_holders, seed=seed, config=_digest({k: v for k, v in config.items() if k not in SPEED_ONLY}), domain_check=domain_check,
                    warmup_epochs=warmup_epochs, clip_feedback=quant, union=union,
                    guide=dict(epochs=guide_epochs, weight=guide_weight) if guide_factory else None,
                    fuzzy=_digest([keywords, fuzzy]) if union == 'fuzzy' else None)

    gen_factory = seeded(gen_factory)                                     # public init
    trunk, spec = flatten(trunk_state(gen_factory(1)))                    # public trunk, independent of U
    local = {}
    for k_, (c, labels) in enumerate(zip(clients, names)):               # own labels, local order
        dev = devices[k_ % len(devices)]
        g = gen_factory(len(labels))
        load_trunk(g, unflatten(trunk, spec))
        gan = ClientCBNGAN(g, disc_factory(len(labels)), config, dev, seed=seed * 100003 + k_)
        shuffle = getattr(c.train_loader, 'sampler', None)
        if hasattr(shuffle, 'generator'):
            shuffle.generator = torch.Generator().manual_seed(seed * 100003 + k_ + 1)
        local[c.id] = dict(gan=gan, loader=c.train_loader, shuffle=shuffle)
    n_cl = len(clients)
    _hashes = {}
    client_hash = lambda k_, c: _hashes.setdefault(k_, data_hash(c.train_loader))
    guide_summary = None
    if guide_factory is not None:                                         # heter: local classifiers
        t = time.perf_counter()
        gcache = Path(generator_cache) if generator_cache else None
        if gcache:
            gcache.mkdir(parents=True, exist_ok=True)
        lr = config.get('local_lr', 1e-3)

        def build(k_, c):                                                 # in order: seeded init
            torch.manual_seed(seed * 100003 + k_ + 7)
            return guide_factory(c.id, len(names[k_]))
        models = [build(k_, c) for k_, c in enumerate(clients)]

        def guide(k_, c):
            gseed, model = seed * 100003 + k_ + 7, models[k_]
            dev = local[c.id]['gan'].device
            acc, hit = None, False
            if gcache:
                key = hashlib.sha256(json.dumps(dict(
                    data=client_hash(k_, c), arch=str(model), labels=len(names[k_]), epochs=guide_epochs, lr=lr,
                    seed=gseed, version=GENERATOR_CACHE_VERSION)).encode()).hexdigest()[:16]
                f = gcache / f'classifier_{key}_epochs{guide_epochs}_seed{seed}.pt'
                lock = _Lock(gcache / f'classifier_{key}.lock')
                while not lock.acquire():                                 # another run trains it
                    if f.exists():
                        break
                    time.sleep(10)
                try:
                    if f.exists():
                        try:
                            st = torch.load(f, map_location='cpu', weights_only=False)
                            if not _finite_state(st['model']):
                                raise ValueError('non-finite weights')
                            model.load_state_dict(st['model'])
                            acc, hit = st['acc'], True
                        except Exception as err:
                            print(f'[heter] {f.name} unusable ({type(err).__name__}: {err}); moved to .bad', flush=True)
                            f.replace(f.with_suffix('.bad'))
                    if not hit:
                        acc = train_guide(model, c.train_loader, guide_epochs, lr, dev, gseed)
                        _save(f, dict(model=model.state_dict(), acc=acc))
                finally:
                    lock.release()
            else:
                acc = train_guide(model, c.train_loader, guide_epochs, lr, dev, gseed)
            local[c.id]['gan'].set_guide(model, guide_weight)
            size = sum(q.numel() for q in model.parameters()) / 1e6
            return f'{type(model).__name__} ({size:.2f}M)', acc, hit

        print(f'[heter] {n_cl} client classifiers x {guide_epochs} epochs on real local data '
              f'(generator loss + {guide_weight} x classifier CE) ...', flush=True)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                got = list(pool.map(lambda kc: guide(*kc), enumerate(clients)))
        else:
            got = [guide(k_, c) for k_, c in enumerate(clients)]
        guide_summary = dict(epochs=guide_epochs, weight=guide_weight, seconds=time.perf_counter() - t,
                             from_cache=sum(h for _, _, h in got), architectures=sorted({a for a, _, _ in got}),
                             train_acc={str(c.id): a for c, (_, a, _) in zip(clients, got)})
        print(f"[heter] classifiers ready: {len(guide_summary['architectures'])} architectures, mean train acc "
              f"{np.mean([a for _, a, _ in got]):.3f}, {guide_summary['from_cache']}/{n_cl} from cache", flush=True)
    shuffle_state = lambda v: getattr(getattr(v['shuffle'], 'generator', None), 'get_state', lambda: None)()
    client_state = lambda v: (lambda st: st if 'shuffle' in st else dict(st, shuffle=shuffle_state(v)))(
        v['gan'].state_dict())                                            # a worker proxy includes its shuffle

    ck = cs = ev_state = None
    if resume:
        ck = _load(resume)
        old = {'warmup_epochs': 0, 'clip_feedback': False, 'union': 'exact', 'fuzzy': None, 'guide': None,
               **ck['manifest']}                                          # older checkpoints
        diff = sorted(k for k in manifest if old.get(k) != manifest[k])
        if diff:
            raise ValueError(f'checkpoint does not match this experiment: {diff} differ')
        base = Path(resume).parent
        cs = _load(base / ck['clients_file'], CHECKPOINT_FORMAT + '/clients')
        ev_state = _load(base / 'evaluator' / 'union.pt', CHECKPOINT_FORMAT + '/evaluator')
        if cs['round'] != ck['round']:
            raise ValueError('client state and Aggregator state are from different rounds')
        Un = dict(cs['union'], U=ck['U'], pks=ck['pks'], info=ck['union_info'],
                  **ev_state['union'])
    warmed_now, warmup_seconds = False, None
    if warmup_epochs and not ck:                                          # local warm-up, before the union
        t = time.perf_counter()
        trunk_id = hashlib.sha256(trunk.tobytes()).hexdigest()
        cache = Path(generator_cache) if generator_cache else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

        def key_of(k_, c):
            gan = local[c.id]['gan']
            return hashlib.sha256(json.dumps(dict(
                data=client_hash(k_, c), labels=len(names[k_]), trunk=trunk_id,
                config=_digest({k: v for k, v in config.items()                # global_*: server classifier
                                if k not in SPEED_ONLY and not k.startswith('global_')}),   # only, not warm-up
                model=str(gan.G) + str(gan.D), seeds=[seed * 100003 + k_, seed * 100003 + k_ + 1],
                version=GENERATOR_CACHE_VERSION,
                **({'guide': dict(arch=str(gan.guide), epochs=guide_epochs, weight=guide_weight)}
                   if gan.guide is not None else {}),
                rng=gan._rng_device.split(':')[0])).encode()).hexdigest()[:16]

        def file_of(key, e):
            return cache / f'generator_{key}_epochs{e}_seed{seed}.pt'

        def warm(k_, c):
            """Returns the epochs loaded from the cache (0: trained from scratch)."""
            v = local[c.id]
            gan = v['gan']
            fresh = copy.deepcopy(gan.state_dict())                           # to undo a failed load
            key = key_of(k_, c) if cache else None
            lock = _Lock(cache / f'generator_{key}.lock') if cache else None
            while lock and not lock.acquire():                            # another run trains this client
                if max(cached_epochs(cache, key, seed), default=0) >= warmup_epochs:
                    break
                time.sleep(10)
            try:
                have = [e for e in (cached_epochs(cache, key, seed) if cache else []) if e <= warmup_epochs]
                done = 0
                for e in sorted(have, reverse=True):                     # newest usable entry
                    f = file_of(key, e)
                    try:
                        st = torch.load(f, map_location='cpu', weights_only=False)
                        if not _finite_state(st['gan']):
                            raise ValueError('non-finite weights')
                        gan.load_state_dict(st['gan'])
                        if st.get('shuffle') is not None and v['shuffle'] is not None:
                            v['shuffle'].generator.set_state(st['shuffle'])
                        done = e
                        break
                    except Exception as err:                             # damaged: set aside, try older
                        print(f'[warm-up] {f.name} unusable ({type(err).__name__}: {err}); moved to .bad',
                              flush=True)
                        f.replace(f.with_suffix('.bad'))
                        gan.load_state_dict(fresh)                       # undo a partial load
                loaded = done
                epochs = gan.epochs
                try:
                    while done < warmup_epochs:
                        step = min(cache_every - done % cache_every, warmup_epochs - done)
                        gan.epochs = step
                        gan.train(v['loader'])
                        done += step
                        if cache:
                            st = gan.state_dict()
                            if not _finite_state(st):
                                raise FloatingPointError(f'client {c.id}: warm-up diverged at epoch {done} '
                                                         '(non-finite weights); nothing cached')
                            _save(file_of(key, done), dict(gan=st, shuffle=shuffle_state(v)))
                            lock.refresh()
                finally:
                    gan.epochs = epochs
                return loaded
            finally:
                if lock:
                    lock.release()

        print(f'[warm-up] {n_cl} clients x {warmup_epochs} local epochs (own labels, before the union)'
              + (f', cache {cache}' if cache else '') + ' ...', flush=True)
        reused = 0
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(warm, k_, c) for k_, c in enumerate(clients)]
                for k_, f in enumerate(as_completed(futures), 1):
                    reused += f.result() == warmup_epochs
                    print(f'[warm-up] {k_}/{n_cl} clients done', flush=True)
        else:
            for k_, c in enumerate(clients):
                reused += warm(k_, c) == warmup_epochs
                print(f'[warm-up] {k_ + 1}/{n_cl} clients done', flush=True)
        if reused:
            print(f'[warm-up] {reused}/{n_cl} clients fully loaded from the cache', flush=True)
        warmed_now, warmup_seconds = True, time.perf_counter() - t
    if not ck:
        samples = ([label_samples(c.train_loader, n, samples_per_label) for c, n in zip(clients, names)]
                   if domain_check and union_result is None else None)
        if union_result is not None:
            Un = union_result
        elif label_psi == 'circuit':
            Un = circuit_union(names, dictionary, ids, keywords if union == 'fuzzy' else None, workers, samples,
                               say, image_match, agg == 'secagg', circuit_tau)
        elif union == 'fuzzy':
            Un = fuzzy_union(names, dictionary, ids, keywords, workers, samples, say, fuzzy, agg == 'secagg',
                             image_match)
        else:
            Un = (cbn_union(names, dictionary, ids, workers, samples, say, image_match) if agg == 'secagg'
                  else plain_union(names, dictionary, ids, samples, say, image_match))
        if agg == 'secagg' and Un['pks'] is None:
            raise ValueError('secagg needs an OPRF union with keys (union_result from cbn_union)')
    U, index = Un['U'], Un['index']
    print(f"[setup] label union ready: {U} labels ({Un['info'].get('method', '?')}, "
          f"{Un['info'].get('seconds', 0):.1f}s); round {1 if not ck else ck['round'] + 1} starts", flush=True)
    if progress:
        say(format_union(Un['view']))
    keys = [[idx[x] for x in l] for idx, l in zip(index, names)]
    predicted, truth = union_relations(ids, names, {k: k for k in range(U)}, keys)
    setup = dict(union=dict(Un['info']), labels=U, generator='cbn', agg=agg, quantize=quant,
                 keep_frac=manifest['keep_frac'], min_holders=min_holders)
    evaluator = dict(union_metrics=Un['metrics'], experimenter_view=Un['view'])
    if checkpoint_dir:
        write_json(Path(checkpoint_dir) / 'setup.json', setup)
        write_json(Path(checkpoint_dir) / 'evaluator' / 'union.json', evaluator)
        if not ck:
            _save(Path(checkpoint_dir) / 'evaluator' / 'union.pt',
                  dict(format=CHECKPOINT_FORMAT + '/evaluator', union=dict(metrics=Un['metrics'], view=Un['view'])))

    g0 = gen_factory(U)
    table = rows(g0)                                                      # U x P, public init
    P, rspec = table.shape[1], row_spec(g0)
    var_mask = np.concatenate([np.full(int(np.prod(s)), n.endswith('running_var')) for n, s, _ in spec])
    for k_, (c, labels) in enumerate(zip(clients, keys)):                # relabel: local a -> union row labels[a]
        v = local[c.id]
        v.update(rows=labels, slots=[index[k_][x] for x in names[k_]],
                 sk=None if Un['sks'] is None else {index[k_][x]: Un['sks'][k_][x] for x in names[k_]})
        if warmed_now:                                                    # keep the warm-up, update vs global init
            v['gan'].rebase(trunk, table[labels])
        else:
            v['gan'].load_global(unflatten(trunk, spec), table[labels])
    if warmup_seconds is not None:
        setup['warmup_seconds'] = warmup_seconds
    if guide_summary is not None:
        setup['guide'] = guide_summary

    blk = lambda sp: np.concatenate([np.full(int(np.prod(sh)), i) for i, (_, sh, _) in enumerate(sp)])
    bt, br = blk(spec), blk(rspec)                                        # tensor-block id per coordinate
    sc_t = compress.initial_scales(spec, trunk, quant_scale0)             # public, per tensor block
    sc_r = np.stack([compress.initial_scales(rspec, table[k], quant_scale0) for k in range(U)])
    mult_t = np.bincount(bt, sc_t) / np.bincount(bt)                      # block scales (clip feedback)
    mult_r = np.bincount(br, sc_r.max(0)) / np.bincount(br)
    qrng = np.random.default_rng(seed)                                    # clients' rounding coins (simulated)
    trainer, history, start = GlobalClassifierTrainer(classifier_factory, config, device), [], 0
    if ck:
        trunk, table, history, start = ck['trunk'], ck['table'], ck['history'], ck['round']
        sc_t, sc_r, mult_t, mult_r = ck['scales']
        if mult_r.size != len(rspec):                                     # checkpoint from one block per row
            mult_r = np.bincount(br, sc_r.max(0)) / np.bincount(br)
        qrng.bit_generator.state = cs['qrng']
        if on_resume:
            on_resume(start)
        for cid, st in cs['clients'].items():
            local[cid]['gan'].load_state_dict(st)
            if st.get('shuffle') is not None:
                local[cid]['shuffle'].generator.set_state(st['shuffle'])
        if ck.get('trainer') is not None:
            trainer.model = trainer.factory(U).to(device)
            trainer.model.load_state_dict(ck['trainer']['model'])
            trainer.optimizer = getattr(torch.optim, config.get('global_model_optim', 'Adam'))(
                trainer.model.parameters(), lr=config.get('global_model_optim_lr', 1e-3))
            trainer.optimizer.load_state_dict(ck['trainer']['optimizer'])
            trainer._mapping = ck['trainer']['mapping']
        _set_rng_state(ck['rng'])
    sent = (ck['sent'] if ck and 'sent' in ck else table).copy()      # the rows as clients hold them
    if workers > 1 and all(str(d) == 'cpu' for d in devices):
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))

    def train_client(c):
        v = local[c.id]
        if comm_only:                                                     # benchmark: same shapes, no training
            rng = np.random.default_rng([seed, c.id, len(history)])
            g = {'T': rng.normal(0, 1e-3, trunk.size)}
            for r_ in set(v['rows']):                                     # own union rows
                g[r_] = rng.normal(0, 1e-3, P)
            return c.id, (g,)
        if procs is not None:                                             # worker: slices before sending back
            v['gan'].keep = keep
        counts = v['gan'].train(v['loader'])
        if not counts:
            return c.id, None
        dt, dr = v['gan'].update(keep)                                    # only the uploaded coordinates
        g, cnt = {'T': dt}, {}
        for a in counts:                                                  # two own labels in one fuzzy
            r_ = v['rows'][a]                                             # group: average their rows
            g[r_], cnt[r_] = g.get(r_, 0) + dr[a], cnt.get(r_, 0) + 1
        return c.id, ({k: (x / cnt[k] if k in cnt else x) for k, x in g.items()},)

    def save_checkpoint(done, warmed=False):
        d = Path(checkpoint_dir)
        cfile = f'clients/round_{done:04d}.pt'
        _save(d / cfile, dict(                                            # 1. clients' private state
            format=CHECKPOINT_FORMAT + '/clients', round=done, qrng=qrng.bit_generator.state,
            union=dict(index=Un['index'], sks=Un['sks']),
            clients={cid: client_state(v) for cid, v in local.items()}))
        agg_state = dict(format=CHECKPOINT_FORMAT, round=done, manifest=manifest, clients_file=cfile,
                         trunk=trunk, table=table, sent=sent, U=U, pks=Un['pks'], union_info=Un['info'],
                         scales=(sc_t, sc_r, mult_t, mult_r), history=history, rng=_rng_state(), warmed=warmed or done > 0,
                         trainer=None if trainer.model is None else dict(
                             model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                             mapping=trainer._mapping))
        _save(d / 'checkpoint_last.pt', agg_state)                        # 2. then the Aggregator's
        if keep_all and done:
            _save(d / f'round_{done:04d}.pt', agg_state)
        elif not keep_all:
            for old_file in (d / 'clients').glob('round_*.pt'):           # 3. drop superseded client files
                if old_file.name != Path(cfile).name:
                    old_file.unlink()

    if warmed_now and checkpoint_dir:
        save_checkpoint(0, warmed=True)                                   # a crash in round 1 keeps it
        write_json(Path(checkpoint_dir) / 'setup.json', setup)

    # clients in worker processes (no GIL; each client's step is launch-bound on one CPU core); same
    # numbers: the pre-decoded images, noise generators and shuffles move with the client
    if client_procs is None:
        client_procs = workers > 1 and n_cl > 1 and all(str(d).startswith('cuda') for d in devices)
    procs = None
    if client_procs and all(hasattr(getattr(v['loader'], 'dataset', None), 'x') for v in local.values()):
        import tempfile
        from secfl.client_procs import ClientPool
        print(f'[clients] {min(workers, n_cl)} worker processes for {n_cl} clients', flush=True)
        try:
            procs = ClientPool(local, clients, min(workers, n_cl),
                               tempfile.mkdtemp(prefix='clients_', dir=checkpoint_dir or None))
        except Exception as e:                                            # clients still here: threads
            print(f'[clients] worker processes failed to start ({e}); using threads', flush=True)
        else:
            for c in clients:
                local[c.id]['gan'] = procs.proxy(c.id, local[c.id]['gan'])
                local[c.id]['loader'] = None                              # the worker has the images (its own
                if not diagnostics:                                       # file): no second copy here
                    c.train_loader = None
            torch.set_num_threads(max(1, os.cpu_count() or 1))           # clients no longer share this process
        if any(str(d).startswith('cuda') for d in devices):
            torch.cuda.empty_cache()
    elif client_procs:
        print('[clients] worker processes need pre-decoded images (tensor loader); using threads', flush=True)

    bb = BulletinBoard()
    model = trainer.model
    diag = None
    if diagnostics and checkpoint_dir and rounds > start:                 # experimenter only (diagnostics.py)
        from diagnostics import Diagnostics
        diag = Diagnostics([c.train_loader for c in clients], ids, truth, predicted,
                           sorted({x for l in names for x in l}), classifier_factory, config, device,
                           Path(checkpoint_dir) / 'diag', {r['cls']: r for r in Un['view']})
    bar = tqdm(range(start, rounds), desc='rounds', unit='round', initial=start, total=rounds, disable=not progress)
    for r in bar:
        row = dict(round=r + 1, seconds={}, bytes=dict(download_per_client=0.))
        t = time.perf_counter()                                           # 3. downlink
        if r:
            from secfl.broadcast import pack_state
            blob = pack_state(unflatten(trunk, spec))
            bb.post('aggregator', f'trunk/{r}', hashlib.sha256(blob).digest())
            if quant:                                                     # 8-bit downlink: every row's change
                d = table - sent                                          # since the copy clients hold, per-row
                sc = (np.maximum(np.abs(d).max(1), 1e-30) / 127).astype(np.float32).astype(np.float64)   # scale
                q = np.round(d / sc[:, None]).astype(np.int8)
                sent = (sent + q * sc[:, None]).astype(np.float32).astype(np.float64)   # = what clients rebuild;
                payload = {g: {'q': np.concatenate([q[g].view(np.uint8), np.float32([sc[g]]).view(np.uint8)])}
                           for g in range(U)}                             # int8 row + float32 scale; rounding error is
            else:                                                         # carried into the next round's delta
                sent, payload = table, {g: {'row': table[g]} for g in range(U)}
            if agg == 'secagg':
                board = BulletinBoard()                                   # this round's KEM posts only
                kem.post_generators(board, Un['pks'], r, payload)
                board_bytes = sum(len(e.payload) for e in board.read())
                row['bytes']['broadcast'] = len(blob) + board_bytes       # one copy
                row['bytes']['download_per_client'] = float(len(blob) + board_bytes)   # everyone reads all
                for c in clients:
                    got, v = kem.fetch_generators(board, local[c.id]['sk'], Un['pks'], r), local[c.id]
                    held = v['gan']._ref[1]                               # own rows of the last round
                    rb = lambda x: x[:-4].view(np.int8) * np.float64(x[-4:].view(np.float32)[0])
                    own = [(held[a] + rb(got[s]['q'])).astype(np.float32) if quant
                           else got[s]['row'] for a, s in enumerate(v['slots'])]
                    v['gan'].load_global(unflatten(trunk, spec), np.stack(own).astype(np.float64))
            else:                                                         # own rows only, same numbers in the clear
                per_row = P + 4 if quant else P * 4
                row['bytes']['broadcast'] = len(blob) + U * per_row
                row['bytes']['download_per_client'] = float(len(blob) + np.mean(
                    [len(local[c.id]['rows']) * per_row for c in clients]))
                for c in clients:
                    local[c.id]['gan'].load_global(unflatten(trunk, spec), sent[local[c.id]['rows']])
        row['seconds']['downlink'] = time.perf_counter() - t

        qr = quantize and not (r == 0 and (warmed_now or bool(ck and ck.get('warmed'))))   # first upload after
        # the local warm-up carries the whole warm-up (far beyond round-sized scales: 8-bit clipping lost up to
        # half of it): sent uncompressed once (plain: floats; secagg: 64-bit fixed point)
        res = max(compress.LEVELS, (1 << 15) // (n_cl + 1))               # clip-rate resolution, no wrap
        it = compress.keep_index(trunk.size, keep_frac if qr else 1., r, b'cbn-trunk')   # public, before
        ir = compress.keep_index(P, keep_frac if qr else 1., r, b'cbn-rows')             # training
        keep = (it, ir)
        t = time.perf_counter()                                           # 4. local training
        inner = tqdm(total=n_cl, desc=f'  round {r + 1} clients', unit='client', leave=False, disable=not progress)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(train_client, c) for c in clients]
                for _ in as_completed(futures):
                    inner.update()
                results = [f.result() for f in futures]
        else:
            results = []
            for c in clients:
                results.append(train_client(c))
                inner.update()
        inner.close()
        if procs is not None:                          # the workers' loaders drew their per-epoch seeds from
            from tensor_loader import _base_seed       # their own global RNG; draw as many here, so the
            for c in clients:                          # Aggregator's later random numbers stay the same
                for _ in range(local[c.id]['gan'].epochs):
                    _base_seed(None)
        row['seconds']['local_training'] = time.perf_counter() - t

        t = time.perf_counter()                                           # 5. aggregation
        if not qr and agg == 'plain':                                  # uncompressed Plain-GeFL
            mt, mr, n = plain_mean(results, U, it, ir)
            up = [4 * sum(g.size for g in u[0].values()) if u else 0 for _, u in results]
            row['bytes'].update(upload=int(sum(up)), upload_payload_per_client=float(np.mean(up)),
                                upload_per_client=float(np.mean(up)))
        else:
            if qr:                                                     # diagnostic: share of clipped values
                over = [(np.abs(_kept(g, it)) > sc_t[it]) if k == 'T' else (np.abs(_kept(g, ir)) > sc_r[k][ir])
                        for _, u in results if u for k, g in u[0].items()]
                row['clipped'] = float(np.concatenate(over).mean()) if over else 0.
            blocks = (bt[it], br[ir], len(spec), len(rspec)) if qr else None
            vectors = {}                                                  # everyone uploads; each client's
            for i, (cid, u) in enumerate(results):                        # float64 update is dropped once
                vectors[cid] = encode_update(u[0] if u else {}, U, it, ir, sc_t, sc_r, qrng, not qr, blocks, res)
                results[i] = (cid, u and ({k: None for k in u[0]},))      # encoded (only its keys are used below)
            if agg == 'secagg':
                sa = {}
                total, _ = run_secagg(vectors, threshold=max(2, -(-2 * n_cl // 3)),
                                      modulus_bits=compress.BITS if qr else 64,
                                      session=f'cbn-round-{r}'.encode(), workers=workers, stats=sa)
            else:                                                         # compressed Plain-GeFL: same
                total = np.sum(np.stack(list(vectors.values())).astype(np.int64), 0) % (1 << compress.BITS)
                # numbers, sent in the clear: 1 byte per 8-bit value, own rows only (+ 2-byte row ids),
                # 1 byte per clip-feedback block
                up = [(it.size if 'T' in u[0] else 0) + sum(ir.size + 2 for k in u[0] if k != 'T')
                      + len(spec) + len(rspec) if u else 0 for _, u in results]
                sa = dict(payload_up=int(sum(up)), control_up=0, control_down=0)
            out = decode_update(total, U, it, ir, sc_t, sc_r, not qr, blocks, res)
            mt, mr, n = out[:3]
            if qr:                                                     # adaptive clipping: each block's
                ct, cr = out[3]                                           # scale follows its clip rate
                step = lambda c: np.where(c > .1, 4., np.where(c > .01, 2., np.where(c < .001, .9, 1.)))
                mult_t, mult_r = mult_t * step(ct), mult_r * step(cr)
                sc_t, sc_r = mult_t[bt], np.broadcast_to(mult_r[br], (U, br.size))   # one row for all U: no U x P copy
                row['clip_feedback'] = dict(trunk_max=float(ct.max()), rows_max=float(cr.max()))
            row['bytes'].update(upload=int(sa['payload_up']), upload_payload_per_client=sa['payload_up'] / n_cl,
                                upload_per_client=(sa['payload_up'] + sa['control_up']) / n_cl)
            row['bytes']['download_per_client'] += sa['control_down'] / n_cl
        trunk, table = trunk.copy(), table.copy()
        if mt is not None and n[0] >= min_holders:
            trunk[it] -= mt                                               # update = -(local - global)
        trunk[var_mask] = np.maximum(trunk[var_mask], 1e-5)               # BN variance stays valid
        for k, m in mr.items():
            if n[k + 1] >= min_holders:
                table[k, ir] -= m
        row['rows_updated'] = int(sum(n[1:] >= min_holders))
        row['rows_below_threshold'] = int(sum((n[1:] > 0) & (n[1:] < min_holders)))
        if not (np.isfinite(trunk).all() and np.isfinite(table).all()):
            raise FloatingPointError(f'round {r + 1}: non-finite global model; last checkpoint is intact')
        row['seconds']['aggregation'] = time.perf_counter() - t
        row['memory'] = _memory(devices)

        if comm_only:
            row.update(accuracy=None, old_acc=None)
            history.append(row)
            if record:
                record(row)
            continue
        t = time.perf_counter()                                           # 6. global classifier
        g = gen_factory(U)
        load_trunk(g, unflatten(trunk, spec))
        set_rows(g, table)
        model = trainer({'all': g.to(device).eval()}, {'all': {k: k for k in range(U)}})
        row['seconds']['global_training'] = time.perf_counter() - t

        t = time.perf_counter()                                           # 7. evaluation
        ev = evaluate_global(model, tests, predicted, truth, device)
        row['seconds']['evaluation'] = time.perf_counter() - t
        row.update(accuracy=ev['ground_truth_acc'], old_acc=ev['old_acc'], evaluation=ev)
        if diag:
            t = time.perf_counter()
            row['diag'] = diag(r + 1, model, g, U, tests)
            row['seconds']['diagnostics'] = time.perf_counter() - t
        bar.set_postfix(acc=f"{row['accuracy']:.4f}")
        history.append(row)
        if record:
            record(row)
        if checkpoint_dir and ((r + 1) % save_every == 0 or r + 1 == rounds):
            t = time.perf_counter()
            save_checkpoint(r + 1)
            row['seconds']['checkpoint'] = time.perf_counter() - t
    bar.close()
    setup.update(trunk_params=int(trunk.size), row_params=int(P), resumed_from=start)
    if procs is not None:
        procs.close()
    return dict(history=history, setup=setup, evaluator=evaluator, model=model, trunk=trunk, table=table, sent=sent, union=Un)
