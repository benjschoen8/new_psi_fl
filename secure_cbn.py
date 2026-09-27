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
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from evaluation import evaluate_global
from label_union import index_metrics
from label_union.oprf_union import oprf_union_with_keys, canonical
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
    """(name, shape, dtype) of every block of a row, in rows() order (for per-tensor scales)."""
    sd = g.state_dict()
    return [(k, (sd[k].shape[1],), np.float32) for k in sorted(sd) if _is_row(k)]


def encode_update(grads, U, it, ir, sc_t, sc_r, rng, fixed=False):
    """grads {'T': trunk update, k: row-k update} (missing = zeros) -> uint64 vector:
    [trunk[it] | row 0[ir] | ... | row U-1[ir] | counts (trunk, rows)].
    fixed=False: clipped 8-bit stochastic rounding, mod 2^16. fixed=True: 2^-24 fixed point, mod 2^64."""
    kt, kr = it.size, ir.size
    v = np.zeros(kt + U * kr + U + 1, np.int64)
    q = ((lambda g, c: np.round(np.clip(g, -CLIP, CLIP) * (1 << FRAC_BITS)).astype(np.int64)) if fixed
         else (lambda g, c: compress.quantize(g, c, rng)))
    for k, g in grads.items():
        if k == 'T':
            v[:kt] = q(g[it], sc_t[it])
            v[kt + U * kr] = 1
        else:
            v[kt + k * kr:kt + (k + 1) * kr] = q(g[ir], sc_r[k][ir])
            v[kt + U * kr + 1 + k] = 1
    return v.view(np.uint64).copy() if fixed else (v % (1 << compress.BITS)).astype(np.uint64)


def decode_update(total, U, it, ir, sc_t, sc_r, fixed=False):
    """SecAgg sum -> (mean trunk update on it or None, {k: mean row-k update on ir}, counts[U+1])."""
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
    t = t_part / n[0] if n[0] else None
    r = {k: r_part[k] / n[k + 1] for k in range(U) if n[k + 1]}
    return t, r, n


def plain_mean(results, U, it, ir):
    """Plain-GeFL: the same contributor mean, in the clear."""
    n, st, sr = np.zeros(U + 1, np.int64), np.zeros(it.size), np.zeros((U, ir.size))
    for _, u in results:
        for k, g in (u[0] if u else {}).items():
            if k == 'T':
                st += g[it]; n[0] += 1
            else:
                sr[k] += g[ir]; n[k + 1] += 1
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


def cbn_union(names, dictionary, ids, workers=1, samples=None, say=lambda *_: None):
    """OPRF union with KEM keys. Returns a dict stored in checkpoints (never recomputed on resume)."""
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


def plain_union(names, dictionary, ids, samples=None, say=lambda *_: None):
    """Plain-GeFL: union of (canonical name, image code) in the clear, no keys."""
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


# ---------------------------------------------------------------------------- run
def run(clients, label_spaces, tests, gen_factory, disc_factory, classifier_factory, config, dictionary,
        agg='secagg', rounds=3, device='cpu', record=None, devices=None, workers=1, checkpoint_dir=None,
        save_every=1, keep_all=False, resume=None, seed=0, progress=False, on_resume=None, domain_check=True,
        samples_per_label=16, union_result=None, quantize=True, keep_frac=0.1, quant_scale0=0.05,
        min_holders=2):
    """gen_factory(num_rows) -> CBN generator; disc_factory(k) -> local D over k labels.
    agg: 'secagg' (the protocol) or 'plain' (Plain-GeFL, no cryptography).
    union_result: reuse a cbn_union() output (the OPRF indices are random per run).
    quantize / keep_frac / quant_scale0: compressed upload (secagg only).
    min_holders: a row is updated only if at least this many clients contributed this round."""
    if agg not in ('plain', 'secagg'):
        raise ValueError('agg must be plain or secagg')
    quant = agg == 'secagg' and quantize
    if quant and len(clients) > MAX_CLIENTS_16BIT:
        raise ValueError(f'16-bit SecAgg holds sums of at most {MAX_CLIENTS_16BIT} clients; use quantize=False')
    devices = list(devices or [device])
    ids = [c.id for c in clients]
    names = [list(label_spaces[i]) for i in ids]
    from tqdm.auto import tqdm
    say = tqdm.write if progress else (lambda *_: None)
    manifest = dict(ids=ids, names=_digest(names), data=[len(c.train_loader.dataset) for c in clients],
                    agg=agg, quantize=quant, keep_frac=keep_frac if quant else 1.0, quant_scale0=quant_scale0,
                    min_holders=min_holders, seed=seed, config=_digest(config), domain_check=domain_check)

    ck = cs = ev_state = None
    if resume:
        ck = _load(resume)
        diff = sorted(k for k in manifest if ck['manifest'].get(k) != manifest[k])
        if diff:
            raise ValueError(f'checkpoint does not match this experiment: {diff} differ')
        base = Path(resume).parent
        cs = _load(base / ck['clients_file'], CHECKPOINT_FORMAT + '/clients')
        ev_state = _load(base / 'evaluator' / 'union.pt', CHECKPOINT_FORMAT + '/evaluator')
        if cs['round'] != ck['round']:
            raise ValueError('client state and Aggregator state are from different rounds')
        Un = dict(cs['union'], U=ck['U'], pks=ck['pks'], info=ck['union_info'], **ev_state['union'])
    else:
        samples = ([label_samples(c.train_loader, n, samples_per_label) for c, n in zip(clients, names)]
                   if domain_check and union_result is None else None)
        Un = union_result or (cbn_union(names, dictionary, ids, workers, samples, say) if agg == 'secagg'
                              else plain_union(names, dictionary, ids, samples, say))
        if agg == 'secagg' and Un['pks'] is None:
            raise ValueError('secagg needs an OPRF union with keys (union_result from cbn_union)')
    U, index = Un['U'], Un['index']
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

    gen_factory = seeded(gen_factory)                                     # public init
    g0 = gen_factory(U)
    trunk, spec = flatten(trunk_state(g0))
    table = rows(g0)                                                      # U x P, public init
    P, rspec = table.shape[1], row_spec(g0)
    var_mask = np.concatenate([np.full(int(np.prod(s)), n.endswith('running_var')) for n, s, _ in spec])
    local = {}
    for k_, (c, labels) in enumerate(zip(clients, keys)):
        dev = devices[k_ % len(devices)]
        g = gen_factory(len(labels))
        gan = ClientCBNGAN(g, disc_factory(len(labels)), config, dev, seed=seed * 100003 + k_)
        gan.load_global(unflatten(trunk, spec), table[labels])
        shuffle = getattr(c.train_loader, 'sampler', None)
        if hasattr(shuffle, 'generator'):
            shuffle.generator = torch.Generator().manual_seed(seed * 100003 + k_ + 1)
        local[c.id] = dict(gan=gan, loader=c.train_loader, shuffle=shuffle, rows=labels,
                           sk=None if Un['sks'] is None else {i: Un['sks'][k_][x] for i, x in zip(labels, names[k_])})

    sc_t = compress.initial_scales(spec, trunk, quant_scale0)             # public, per tensor
    sc_r = np.stack([compress.initial_scales(rspec, table[k], quant_scale0) for k in range(U)])
    qrng = np.random.default_rng(seed)                                    # clients' rounding coins (simulated)
    trainer, history, start = GlobalClassifierTrainer(classifier_factory, config, device), [], 0
    if ck:
        trunk, table, history, start = ck['trunk'], ck['table'], ck['history'], ck['round']
        sc_t, sc_r = ck['scales']
        qrng.bit_generator.state = cs['qrng']
        if on_resume:
            on_resume(start)
        for cid, st in cs['clients'].items():
            local[cid]['gan'].load_state_dict(st)
            if st.get('shuffle') is not None:
                local[cid]['shuffle'].generator.set_state(st['shuffle'])
        if ck['trainer'] is not None:
            trainer.model = trainer.factory(U).to(device)
            trainer.model.load_state_dict(ck['trainer']['model'])
            trainer.optimizer = getattr(torch.optim, config.get('global_model_optim', 'Adam'))(
                trainer.model.parameters(), lr=config.get('global_model_optim_lr', 1e-3))
            trainer.optimizer.load_state_dict(ck['trainer']['optimizer'])
            trainer._mapping = ck['trainer']['mapping']
        _set_rng_state(ck['rng'])
    if workers > 1 and all(str(d) == 'cpu' for d in devices):
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))

    def train_client(c):
        v = local[c.id]
        counts = v['gan'].train(v['loader'])
        if not counts:
            return c.id, None
        dt, dr = v['gan'].update()
        return c.id, ({'T': dt, **{v['rows'][a]: dr[a] for a in counts}},)

    bb = BulletinBoard()
    model = trainer.model
    n_cl = len(clients)
    bar = tqdm(range(start, rounds), desc='rounds', unit='round', initial=start, total=rounds, disable=not progress)
    for r in bar:
        row = dict(round=r + 1, seconds={}, bytes=dict(download_per_client=0.))
        t = time.perf_counter()                                           # 3. downlink
        if r:
            from secfl.broadcast import pack_state
            blob = pack_state(unflatten(trunk, spec))
            bb.post('aggregator', f'trunk/{r}', hashlib.sha256(blob).digest())
            if agg == 'secagg':
                board = BulletinBoard()                                   # this round's KEM posts only
                kem.post_generators(board, Un['pks'], r, {k: {'row': table[k]} for k in range(U)})
                board_bytes = sum(len(e.payload) for e in board.read())
                row['bytes']['broadcast'] = len(blob) + board_bytes       # one copy
                row['bytes']['download_per_client'] = float(len(blob) + board_bytes)   # everyone reads all
                for c in clients:
                    got = kem.fetch_generators(board, local[c.id]['sk'], Un['pks'], r)
                    local[c.id]['gan'].load_global(unflatten(trunk, spec),
                                                   np.stack([got[i]['row'] for i in local[c.id]['rows']]))
            else:
                row['bytes']['broadcast'] = len(blob) + table.astype(np.float32).nbytes
                row['bytes']['download_per_client'] = float(len(blob) + np.mean(
                    [len(local[c.id]['rows']) * P * 4 for c in clients]))
                for c in clients:
                    local[c.id]['gan'].load_global(unflatten(trunk, spec), table[local[c.id]['rows']])
        row['seconds']['downlink'] = time.perf_counter() - t

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
        row['seconds']['local_training'] = time.perf_counter() - t

        t = time.perf_counter()                                           # 5. aggregation
        it = compress.keep_index(trunk.size, keep_frac if quant else 1., r, b'cbn-trunk')
        ir = compress.keep_index(P, keep_frac if quant else 1., r, b'cbn-rows')
        if agg == 'plain':
            mt, mr, n = plain_mean(results, U, it, ir)
            up = [4 * sum(g.size for g in u[0].values()) if u else 0 for _, u in results]
            row['bytes'].update(upload=int(sum(up)), upload_payload_per_client=float(np.mean(up)),
                                upload_per_client=float(np.mean(up)))
        else:
            vectors = {cid: encode_update(u[0] if u else {}, U, it, ir, sc_t, sc_r, qrng, fixed=not quant)
                       for cid, u in results}                             # everyone uploads
            sa = {}
            total, _ = run_secagg(vectors, threshold=max(2, -(-2 * n_cl // 3)),
                                  modulus_bits=compress.BITS if quant else 64,
                                  session=f'cbn-round-{r}'.encode(), workers=workers, stats=sa)
            mt, mr, n = decode_update(total, U, it, ir, sc_t, sc_r, fixed=not quant)
            row['bytes'].update(upload=int(sa['payload_up']), upload_payload_per_client=sa['payload_up'] / n_cl,
                                upload_per_client=(sa['payload_up'] + sa['control_up']) / n_cl)
            row['bytes']['download_per_client'] += sa['control_down'] / n_cl
        trunk, table = trunk.copy(), table.copy()
        if mt is not None and n[0] >= min_holders:
            trunk[it] -= mt                                               # update = -(local - global)
            d = np.zeros(trunk.size); d[it] = -mt
            sc_t = compress.next_scales(spec, d, it)
        trunk[var_mask] = np.maximum(trunk[var_mask], 1e-5)               # BN variance stays valid
        for k, m in mr.items():
            if n[k + 1] >= min_holders:
                table[k, ir] -= m
                d = np.zeros(P); d[ir] = -m
                sc_r[k] = compress.next_scales(rspec, d, ir)
        row['rows_updated'] = int(sum(n[1:] >= min_holders))
        row['rows_below_threshold'] = int(sum((n[1:] > 0) & (n[1:] < min_holders)))
        if not (np.isfinite(trunk).all() and np.isfinite(table).all()):
            raise FloatingPointError(f'round {r + 1}: non-finite global model; last checkpoint is intact')
        row['seconds']['aggregation'] = time.perf_counter() - t

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
        bar.set_postfix(acc=f"{row['accuracy']:.4f}")
        history.append(row)
        if record:
            record(row)
        if checkpoint_dir and ((r + 1) % save_every == 0 or r + 1 == rounds):
            t = time.perf_counter()
            d = Path(checkpoint_dir)
            cfile = f'clients/round_{r + 1:04d}.pt'
            _save(d / cfile, dict(                                        # 1. clients' private state
                format=CHECKPOINT_FORMAT + '/clients', round=r + 1, qrng=qrng.bit_generator.state,
                union=dict(index=Un['index'], sks=Un['sks']),
                clients={cid: dict(v['gan'].state_dict(), shuffle=getattr(getattr(v['shuffle'], 'generator', None),
                                                                           'get_state', lambda: None)())
                         for cid, v in local.items()}))
            agg_state = dict(format=CHECKPOINT_FORMAT, round=r + 1, manifest=manifest, clients_file=cfile,
                             trunk=trunk, table=table, U=U, pks=Un['pks'], union_info=Un['info'],
                             scales=(sc_t, sc_r), history=history, rng=_rng_state(),
                             trainer=dict(model=model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                                          mapping=trainer._mapping))
            _save(d / 'checkpoint_last.pt', agg_state)                    # 2. then the Aggregator's
            if keep_all:
                _save(d / f'round_{r + 1:04d}.pt', agg_state)
            else:
                for old in (d / 'clients').glob('round_*.pt'):            # 3. drop superseded client files
                    if old.name != Path(cfile).name:
                        old.unlink()
            row['seconds']['checkpoint'] = time.perf_counter() - t
    bar.close()
    setup.update(trunk_params=int(trunk.size), row_params=int(P), resumed_from=start)
    return dict(history=history, setup=setup, evaluator=evaluator, model=model, trunk=trunk, table=table, union=Un)
