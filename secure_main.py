"""Secure pull protocol, end to end (exact labels):

  setup   union + keys, no Aggregator input needed:
            --union oprf  (default) label_union.oprf_union_with_keys: n-party OPRF tags + SecAgg;
                          sk = KDF(tag), so only holders of a label can derive its key; the pks
                          reach the Aggregator through one more SecAgg (no sender, no counts);
                          optional weak image check (label_union.domain)
            --union mpc   label_union.mpc_union in MPC (slow beyond a few clients)
            --union ideal the same outputs computed in the clear (simulation only)
          -> every client knows slot(label) and sk_slot; pk_0..pk_{M-1} on the BB
  round   1. Aggregator KEM-posts G_s for every slot s < M (secfl.kem; inactive = zeros)
          2. clients decrypt their own slots, train one GAN per held label (secfl.label_gan)
          3. upload -(theta_local - theta_global) and n per slot:
               --agg plain  : sent in the clear (baseline)
               --agg secagg : fixed-length vector over all M slots, zero for non-held
                              (secfl.upload) through Bonawitz SecAgg (secfl.secagg)
          4. theta_s <- sample-weighted FedAvg if |Group_s| >= t (secfl.settle, lr=1)
          5. Aggregator trains the global classifier from the active slots' generators
          6. evaluation on every client's test split (labels mapped through slots)

Run from the project root:
  python -m secure_main --smoke --agg secagg --union mpc --rounds 3
  python -m secure_main --device cuda:0 --agg secagg --union ideal --rounds 20
"""
import json
import secrets
import time
from pathlib import Path

import numpy as np
import torch

from secfl import kem, ristretto as rg
from secfl.bb import BulletinBoard, PublicParams
from secfl.label_gan import ClientLabelGANs, pair_state, load_pair_state, as_trainer_inputs
from secfl.secagg import run_secagg
from secfl.settle import flatten, unflatten, settle
from secfl.upload import UploadLayout
from training import GlobalClassifierTrainer


# ---------------------------------------------------------------------------- setup
def ideal_union_with_keys(client_labels, m_max, hide_count=True, max_labels=None):
    """Same outputs as label_union.mpc_union.exact_union_with_keys, computed in the clear (for scale)."""
    n_rows = 1 << int(np.ceil(np.log2(len(client_labels) * m_max)))
    if max_labels is not None:
        n_rows = min(n_rows, 1 << int(np.ceil(np.log2(max_labels))))
    names = sorted({x for labels in client_labels for x in labels})
    if len(names) > n_rows:
        raise ValueError(f'union larger than the public slot bound M={n_rows}')
    order = secrets.SystemRandom().sample(range(n_rows), len(names)) if hide_count else range(len(names))
    slot_of = dict(zip(names, order))
    sks = [rg.random_scalar() for _ in range(n_rows)]
    pks = [(rg.BASE * sk).encode() for sk in sks]
    slots = [{x: slot_of[x] for x in labels} for labels in client_labels]
    keys = [{x: sks[slot_of[x]] for x in labels} for labels in client_labels]
    return slots, keys, pks, dict(rows=n_rows, seconds=0.0, and_gates=0)


def setup_union(client_labels, m_max, backend='oprf', hide_count=True, max_labels=None, domains=None, workers=1):
    if backend == 'oprf':                  # indices are dense 0..U-1: the Aggregator learns U
        from label_union.oprf_union import oprf_union_with_keys
        return oprf_union_with_keys(client_labels, workers=workers, domains=domains)
    if backend == 'ideal':
        return ideal_union_with_keys(client_labels, m_max, hide_count, max_labels)
    from align.mpc import MPC
    from label_union.mpc_union import exact_union_with_keys
    return exact_union_with_keys(client_labels, m_max, MPC(len(client_labels), b'secure-main'), hide_count, max_labels)


def aggregate(updates, layout=None, session=b'', workers=1):
    """updates {client: ({slot: -delta}, {slot: n})} -> (sum n*g, N, |Group|, upload bytes).

    layout None: plaintext baseline. Otherwise every client uploads the fixed-length vector
    over all M slots through SecAgg (threshold ceil(2n/3)); the server sees only the sum.
    """
    if layout is None:
        grad_sums, N, G = {}, {}, {}
        for grads, counts in updates.values():
            for s, g in grads.items():
                grad_sums[s] = grad_sums.get(s, 0) + counts[s] * g
                N[s] = N.get(s, 0) + counts[s]
                G[s] = G.get(s, 0) + 1
        return grad_sums, N, G, sum(g.size * 8 for grads, _ in updates.values() for g in grads.values())
    vectors = {cid: layout.encode(*u) for cid, u in updates.items()}
    total, _ = run_secagg(vectors, threshold=max(2, -(-2 * len(vectors) // 3)),
                          modulus_bits=layout.k, session=session, workers=workers)
    grad_sums, N, G = layout.decode(total)
    return grad_sums, N, G, len(vectors) * layout.size * 8


# ---------------------------------------------------------------------------- one run
def run(clients, label_spaces, tests, gen_factory, disc_factory, classifier_factory, config,
        agg='secagg', union='oprf', hide_count=True, rounds=3, threshold=1, device='cpu', record=None,
        max_labels=None, domain_check=True, samples_per_label=16, workers=1, share_d=False, quantize=True,
        keep_frac=0.1, quant_scale0=0.05, seed=0):
    """clients: objects with .id and .train_loader; label_spaces {id: ordered names}.

    Upload size (--agg secagg), see secfl.compress: share_d=False keeps every D local (only G is
    aggregated and broadcast); quantize=True sends 8-bit stochastic-rounded updates through a
    16-bit SecAgg; keep_frac < 1 uploads only a public random subset of coordinates per round.
    All slots start from one public init (seeded), so the uploads are small deltas."""
    if agg not in ('plain', 'secagg'):
        raise ValueError('agg must be plain or secagg')
    ids = [c.id for c in clients]
    names = [list(label_spaces[i]) for i in ids]
    m_max = max(len(x) for x in names)
    t0 = time.perf_counter()
    domains = None
    if union == 'oprf' and domain_check:                                  # weak image check, client-local
        from secure_code_no_cluster import label_samples
        from label_union.domain import client_codes
        samples = [label_samples(c.train_loader, n, samples_per_label) for c, n in zip(clients, names)]
        domains = [client_codes(s, l)[0] for s, l in zip(samples, names)]
    slots, keys, pks, ustats = setup_union(names, m_max, union, hide_count, max_labels, domains, workers)
    M = len(pks)
    setup_seconds = time.perf_counter() - t0

    bb = BulletinBoard()
    for pk_index, pk in enumerate(pks):
        bb.post('setup', f'pk/{pk_index}', pk)
    from secure_code_no_cluster import seeded
    gen_factory = seeded(gen_factory)                                     # public init, same for every slot
    local = {}
    for c, labels, sl, ks in zip(clients, names, slots, keys):
        local[c.id] = dict(
            gans=ClientLabelGANs({a: sl[x] for a, x in enumerate(labels)}, gen_factory, disc_factory, config, device,
                                 share_d=share_d),
            sk={sl[x]: ks[x] for x in labels}, slot_of_local={a: sl[x] for a, x in enumerate(labels)},
            loader=c.train_loader)

    template = pair_state(gen_factory(), disc_factory(), include_d=share_d)
    flat0, spec = flatten(template)
    thetas = {s: flat0.copy() for s in range(M)}                          # public init for every slot
    for c in clients:                                                     # everyone knows the public init
        local[c.id]['gans'].load_global({s: template for s in local[c.id]['sk']})
    from secfl import compress
    scales = {s: compress.initial_scales(spec, flat0, quant_scale0) for s in range(M)}   # public, per slot and tensor
    rng = np.random.default_rng(seed)
    active, history = set(), []
    trainer = GlobalClassifierTrainer(classifier_factory, config, device)
    n_max = max(len(c.train_loader.dataset) for c in clients)
    layout = None
    if agg == 'secagg' and not quantize:
        params = PublicParams(labels=tuple(range(M)), modulus_bits=64, frac_bits=24,
                              clip=config.get('upload_clip', 1e3), threshold=threshold)
        layout = UploadLayout(params, {s: flat0.size for s in range(M)}, n_max, len(clients))

    for r in range(rounds):
        row = dict(round=r + 1, seconds={}, bytes={})
        t = time.perf_counter()
        if active:                                                            # 1. KEM downlink
            kem.post_generators(bb, pks, r, {s: unflatten(thetas[s], spec) for s in active})
            row['bytes']['broadcast'] = sum(len(e.payload) for e in bb.read() if e.topic.startswith(f'gen/{r}/'))
            for c in clients:
                got = kem.fetch_generators(bb, local[c.id]['sk'], pks, r)
                local[c.id]['gans'].load_global(got)
        row['seconds']['downlink'] = time.perf_counter() - t

        t = time.perf_counter()                                               # 2. local training
        updates = {}
        for c in clients:
            gans = local[c.id]['gans']
            updates[c.id] = gans.updates(gans.train(local[c.id]['loader']))
        row['seconds']['local_training'] = time.perf_counter() - t

        t = time.perf_counter()                                               # 3. aggregation
        if agg == 'secagg' and quantize:
            idx = compress.keep_index(flat0.size, keep_frac, r)
            k = idx.size
            kept_scales = {s_: sc[idx] for s_, sc in scales.items()}
            vectors = {cid: compress.encode({s: g[idx] for s, g in grads.items()}, M, k, kept_scales, rng)
                       for cid, (grads, _) in updates.items()}
            total, _ = run_secagg(vectors, threshold=max(2, -(-2 * len(vectors) // 3)),
                                  modulus_bits=compress.BITS, session=f'round-{r}'.encode())
            means, holders = compress.decode(total, M, k, kept_scales)
            updated = []
            for s_, mean in means.items():
                if holders[s_] >= threshold:
                    thetas[s_] = thetas[s_].copy()
                    thetas[s_][idx] -= mean                                    # update = -(local - global)
                    delta = np.zeros(flat0.size)
                    delta[idx] = -mean
                    scales[s_] = compress.next_scales(spec, delta, idx)
                    updated.append(s_)
            row['bytes']['upload'] = len(vectors) * (M * k + M) * compress.BITS // 8
        else:
            grad_sums, N, G, row['bytes']['upload'] = aggregate(updates, layout, f'round-{r}'.encode())
            thetas, updated = settle(thetas, grad_sums, N, G, lr=1.0, threshold=threshold)
        active |= set(updated)
        row['seconds']['aggregation'] = time.perf_counter() - t
        row['active_slots'] = len(active)

        if not active:                                                        # nothing reached t yet
            row['accuracy'] = None
            history.append(row)
            if record:
                record(row)
            continue
        t = time.perf_counter()                                               # 5. global classifier
        dense = {s: k for k, s in enumerate(sorted(active))}
        generators = {}
        for s, k in dense.items():
            g, d = gen_factory(), disc_factory()
            load_pair_state(g, d, unflatten(thetas[s], spec))
            generators[k] = g.to(device).eval()
        model = trainer(*as_trainer_inputs(generators))
        row['seconds']['global_training'] = time.perf_counter() - t

        correct = total_n = 0                                                 # 6. evaluation
        model.eval()
        with torch.no_grad():
            for _, cid, loader in tests:
                to_class = local[cid]['slot_of_local']
                for x, y in loader:
                    out = model(x.to(device))
                    pred = (out[1] if isinstance(out, tuple) else out).argmax(1).cpu()
                    target = torch.tensor([dense.get(to_class[int(v)], -1) for v in y])
                    correct += int((pred == target).sum())
                    total_n += len(y)
        row['accuracy'] = correct / max(1, total_n)
        history.append(row)
        if record:
            record(row)
    return dict(history=history, slots=slots, setup=dict(ustats, seconds=setup_seconds, slots=M),
                model=model if active else None)


# ---------------------------------------------------------------------------- CLI
def main():
    from setup import parser as base_parser, seed_all, build_clients
    from omegaconf import OmegaConf
    p = base_parser()
    p.description = 'Secure pull protocol (exact union + KEM + SecAgg)'
    p.add_argument('--agg', choices=('plain', 'secagg'), default='secagg')
    p.add_argument('--union', choices=('oprf', 'mpc', 'ideal'), default='oprf',
                   help='oprf: n-party OPRF + SecAgg (default); mpc: label_union.mpc_union (slow beyond a few '
                        'clients); ideal: same outputs computed in the clear (simulation only)')
    p.add_argument('--no-domain-check', action='store_true', help='oprf union: names only, no image check')
    p.add_argument('--samples-per-label', type=int, default=16, help='images per label for the domain check')
    p.add_argument('--workers', type=int, default=1, help='processes for the OPRF ring')
    p.add_argument('--share-d', action='store_true', help='also aggregate D (default: D stays local)')
    p.add_argument('--no-quantize', action='store_true', help='64-bit fixed point SecAgg instead of 8-bit / 16-bit')
    p.add_argument('--keep-frac', type=float, default=0.1, help='fraction of coordinates uploaded per round')
    p.add_argument('--quant-scale0', type=float, default=0.05, help='first-round quantization scale')
    p.add_argument('--dense-slots', action='store_true', help='do not hide the union size')
    p.add_argument('--threshold', type=int, default=1, help='t: min holders before a slot updates')
    p.add_argument('--max-labels', type=int, help='public bound M on distinct labels (slot space)')
    args = p.parse_args()
    from setup import resolve_device
    args.device = resolve_device(args.device)
    seed_all(args.seed)
    config = OmegaConf.to_container(OmegaConf.load(args.exp_conf), resolve=True)
    if args.smoke:
        from smoke import TinyGenerator, TinyDiscriminator, TinyClassifier
        config.update(gen_local_epochs=1, global_model_epochs=1, batch_size=4,
                      global_samples_per_class=4, gen_noise_dim=4)
        clients, spaces, tests = build_synthetic(config)
        gen_f, disc_f, cls_f = (lambda: TinyGenerator(1)), TinyDiscriminator, TinyClassifier
    else:
        from nets import DCGANGenerator, DCGANDiscriminator
        clients, spaces, tests, _, _, cls_f = build_clients(args, config)
        nd = config.get('gen_noise_dim', 128)
        gen_f, disc_f = (lambda: DCGANGenerator(1, noise_dim=nd)), (lambda: DCGANDiscriminator(1))
    out = args.output or Path('runs') / (time.strftime('%Y%m%dT%H%M%S') + f'_secure_{args.agg}')
    out.mkdir(parents=True, exist_ok=False)

    def record(row):
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        acc = 'pending' if row['accuracy'] is None else f"{row['accuracy']:.4f}"
        print(f"Round {row['round']}: acc={acc} active={row['active_slots']} "
              f"upload={row['bytes'].get('upload', 0)}B")
    result = run(clients, spaces, tests, gen_f, disc_f, cls_f, config, agg=args.agg, union=args.union,
                 hide_count=not args.dense_slots, rounds=args.rounds or config.get('global_rounds', 45),
                 threshold=args.threshold, device=args.device, record=record, max_labels=args.max_labels,
                 domain_check=not args.no_domain_check, samples_per_label=args.samples_per_label,
                 workers=args.workers, share_d=args.share_d, quantize=not args.no_quantize,
                 keep_frac=args.keep_frac, quant_scale0=args.quant_scale0, seed=args.seed)
    (out / 'setup.json').write_text(json.dumps(result['setup'], indent=2, default=str))
    print(f'Artifacts: {out.resolve()}')


def build_synthetic(config, label_sets=(('cat', 'dog'), ('dog', 'ship'), ('ship', 'cat'))):
    """Tiny 2x2 RGB images; each label name has its own brightness pattern. CPU seconds."""
    from torch.utils.data import DataLoader, TensorDataset
    from types import SimpleNamespace
    base = {name: (k * 2 - 2) * .4 for k, name in enumerate(sorted({x for s in label_sets for x in s}))}
    clients, spaces, tests = [], {}, []
    for cid, labels in enumerate(label_sets):
        xs, ys = [], []
        for a, name in enumerate(labels):
            xs.append(torch.full((8, 3, 2, 2), base[name]).clamp(-1, 1) + .05 * torch.randn(8, 3, 2, 2))
            ys.append(torch.full((8,), a))
        x, y = torch.cat(xs), torch.cat(ys)
        clients.append(SimpleNamespace(id=cid, train_loader=DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)))
        spaces[cid] = labels
        tests.append(('synthetic', cid, DataLoader(TensorDataset(x, y), batch_size=8)))
    return clients, spaces, tests


if __name__ == '__main__':
    main()
