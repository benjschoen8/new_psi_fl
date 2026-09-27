"""Secure pull protocol, PRIVATE-PACFL version: clients group themselves before training, then
each group trains its own shared-trunk conditional GAN (public label codes, exact labels).

Preprocessing (clients only, no Aggregator):
  1. U_i = PACFL basis of client i's data after a public random projection (align.pacfl_plain)
  2. every pair: OLE + 2PC -> XOR shares of [s_ij > tau] (align.pacfl_similarity)
  3. n-party MPC: connected components -> each client learns its group members and slot;
     everyone learns the group count G; the Aggregator receives only the per-group label
     existence bits (align.private_clusters). Group sizes and memberships never leave the clients.
Every round:
  4. Aggregator broadcasts the G trunks; each client loads the one of its slot
  5. clients train G(z, c_L) on their own labels with a private local D
  6. client uploads a G x |trunk| vector: -(theta_local - theta_global) / |group| in its slot,
     zeros elsewhere, through SecAgg (--agg secagg) or in the clear (--agg plain). The sum per
     slot is the group mean, so the Aggregator needs no counts: it never learns group sizes
  7. Aggregator trains the global classifier on G_s(z, c_L) for every label group s holds
  8. evaluation on every client's test split

--cluster private (the protocol) | plain (same grouping in the clear, reference)
          | original (PACFL principal angles + hierarchical clustering, in the clear)
          | none (one group of everyone: the no-cluster baseline on the same pipeline)

  python -m secure_code_cluster --smoke --agg secagg --rounds 3
  python -m secure_code_cluster --devices cuda:0,cuda:1 --workers 4 --tau .5 --rounds 45
  python -m secure_code_cluster --resume runs/<run>/checkpoint_last.pt --rounds 60
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from align.pacfl_plain import projection_matrix, projected_basis, plain_groups, original_pacfl, agreement
from align.pacfl_similarity import all_pairs
from align.private_clusters import private_clusters, from_groups
from secfl.bb import BulletinBoard
from secfl.code_gan import ClientCodeGAN, generator_state, load_generator_state
from label_union import indicator
from secfl.label_codes import label_code, index_code
from secfl.secagg import run_secagg
from secfl.settle import flatten, unflatten
from evaluation import evaluate_global
from secure_code_no_cluster import (seeded, _rng_state, _set_rng_state, TinyCodeGenerator, TinyLocalDiscriminator,
                                    union_relations, write_json, trim_log, subsample, build_union, format_union,
                                    label_samples)
from training import GlobalClassifierTrainer

CHECKPOINT_FORMAT = 'secure_code_cluster.v1'
FRAC = 24                                   # fixed-point bits of uploads in Z_2^64


def save_checkpoint(path, **state):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(format=CHECKPOINT_FORMAT, **state), tmp)
    tmp.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)      # own files only
    if state.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{path}: not a {CHECKPOINT_FORMAT} checkpoint')
    return state


# ---------------------------------------------------------------------------- preprocessing
def input_dim(loader):
    x, _ = next(iter(loader))
    return int(np.prod(x.shape[1:]))


def make_groups(clients, keys, table, cluster='private', tau=.5, proj_dim=128, budget=20, alpha=20,
                workers=1, say=lambda *_: None):
    """Returns (grouping dict as align.private_clusters, stats, experimenter-only info)."""
    ind = np.stack([indicator(k, table) for k in keys]).astype(np.uint8)          # over the union table
    dims = {input_dim(c.train_loader) for c in clients}
    if len(dims) != 1:
        raise ValueError(f'clients have different input sizes {dims}; the projection needs one')
    d = dims.pop()
    R = projection_matrix(d, min(proj_dim, d))                          # public seed
    t = time.perf_counter()
    bases = [projected_basis(c.train_loader, R, budget=budget) for c in clients]  # each client, locally
    stats = dict(basis_seconds=time.perf_counter() - t, proj_dim=R.shape[1], input_dim=d, tau=tau)
    plain, S = plain_groups(bases, tau)
    if cluster == 'private':
        say(f'[setup] private PACFL: {len(clients) * (len(clients) - 1) // 2} pairwise OLE+2PC ...')
        t = time.perf_counter()
        E, pair_stats = all_pairs(bases, tau, workers=workers)
        stats.update(pairs_seconds=time.perf_counter() - t, **{f'pairs_{k}': v for k, v in pair_stats.items()})
        say(f'[setup] pairs done in {stats["pairs_seconds"]:.1f}s; {len(clients)}-party components MPC ...')
        t = time.perf_counter()
        out = private_clusters(E, ind, workers=workers)
        stats.update(mpc_seconds=time.perf_counter() - t, **{f'mpc_{k}': v for k, v in out.pop('stats').items()})
    elif cluster == 'plain':
        out = from_groups(plain, ind)
    elif cluster == 'none':                                         # baseline: one group of everyone
        out = from_groups([list(range(len(clients)))], ind)
    elif cluster == 'original':
        from clustering import local_basis
        out = from_groups(original_pacfl([local_basis(c.train_loader, budget) for c in clients], alpha), ind)
    else:
        raise ValueError('cluster must be private, plain, original or none')
    groups = sorted({tuple(c['members']) for c in out['clients']})
    view = dict(groups=[list(g) for g in groups], similarity=S.round(4).tolist(),
                agreement_with_plain=agreement([list(g) for g in groups], plain, len(clients)))
    return out, stats, view


# ---------------------------------------------------------------------------- uploads
def encode_upload(update, slot, G, clip, scale):
    """G x d vector over Z_2^64: clip(update) * scale in the client's slot, zeros elsewhere."""
    u = np.asarray(update, np.float64)
    norm = np.linalg.norm(u)
    u = u * min(1.0, clip / norm) * scale if norm > 0 else u
    vec = np.zeros((G, u.size), np.uint64)
    vec[slot] = np.rint(u * 2 ** FRAC).astype(np.int64).astype(np.uint64)
    return vec.ravel()


def decode_sum(total, G):
    return np.asarray(total, np.uint64).view(np.int64).reshape(G, -1) / 2 ** FRAC


class GroupCoded(nn.Module):
    """Group trunk with the group's labels as local ids 0..k-1 (GlobalClassifierTrainer input)."""
    def __init__(self, g, codes):
        super().__init__()
        self.g, self.C = g, torch.as_tensor(np.stack(codes))

    def forward(self, z, y):
        return self.g(z, self.C.to(z.device)[y])


# ---------------------------------------------------------------------------- run
def run(clients, label_spaces, tests, gen_factory, disc_factory, classifier_factory, config, dictionary,
        cluster='private', tau=.5, proj_dim=128, basis_budget=20, alpha=20, agg='secagg', rounds=3, code_dim=128,
        device='cpu', record=None, devices=None, workers=1, checkpoint_dir=None, save_every=1, keep_all=False,
        resume=None, seed=0, progress=False, on_resume=None, union='oprf',
        domain_check=True, samples_per_label=16):
    """As secure_code_no_cluster.run, plus the grouping (cluster, tau, proj_dim, basis_budget, alpha).

    The label union runs first (label_union, --union); the groups' label sets are then over its keys:
    indices (oprf / mpc: the Aggregator sees only which indices each group has) or names (secagg)."""
    if agg not in ('plain', 'secagg'):
        raise ValueError('agg must be plain or secagg')
    devices = list(devices or [device])
    ids = [c.id for c in clients]
    names = [list(label_spaces[i]) for i in ids]
    from tqdm.auto import tqdm
    say = tqdm.write if progress else (lambda *_: None)

    ck = load_checkpoint(resume) if resume else None
    if ck:                                                             # union + grouping are one-time: reuse
        if (ck['cluster'] != cluster or ck.get('union', {}).get('info', {}).get('method') != union or ck['agg'] != agg
                or ck['code_dim'] != code_dim or ck['clients'].keys() != set(ids)):
            raise ValueError('checkpoint does not match this experiment (cluster, union, agg, code_dim or clients)')
        U, grouping, group_stats, view = ck['union'], ck['grouping'], ck['group_stats'], ck['view']
    else:
        samples = ([label_samples(c.train_loader, n, samples_per_label) for c, n in zip(clients, names)]
                   if union == 'oprf' and domain_check else None)
        U = build_union(names, dictionary, ids, union, workers, say, samples)
        grouping, group_stats, view = make_groups(clients, U['keys'], U['existing'], cluster, tau, proj_dim,
                                                  basis_budget, alpha, workers, say)
    existing, keys, union_metrics = U['existing'], U['keys'], U['metrics']
    code = label_code if union == 'secagg' else index_code
    if progress:
        say(format_union(U['view']))
    G, slots = grouping['groups'], [c['slot'] for c in grouping['clients']]
    sizes = [len(c['members']) for c in grouping['clients']]          # each client knows only its own
    held = [[x for x, h in zip(existing, row) if h] for row in grouping['labels']]    # Aggregator's view
    class_of = {x: k for k, x in enumerate(existing)}
    predicted, truth = union_relations(ids, names, class_of, keys)       # for global accuracy
    setup = dict(cluster=cluster, groups=G, aggregator_view=dict(zip(range(G), held)), labels=len(existing),
                 union=U['info'], union_metrics=union_metrics, grouping_stats=group_stats, experimenter_view=view,
                 union_view=U['view'])
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(checkpoint_dir) / 'setup.json', setup)          # early: a crashed run still has it
    say(f'[setup] {G} groups; Aggregator sees label sets per group: {held}')
    if progress:
        say(f'[experimenter view, no party sees this] groups={view["groups"]} '
            f'ARI vs plain grouping={view["agreement_with_plain"]:.3f}')

    gen_factory = seeded(gen_factory)
    theta0, spec = flatten(generator_state(gen_factory()))
    thetas = np.tile(theta0, (G, 1))                                   # same public init for every group
    local = {}
    for k, (c, labels) in enumerate(zip(clients, keys)):
        dev = devices[k % len(devices)]
        codes = {a: code(x, code_dim) for a, x in enumerate(labels)}
        gan = ClientCodeGAN(codes, gen_factory(), disc_factory(len(labels)), config, dev, seed=seed * 100003 + k)
        gan.load_global(unflatten(thetas[slots[k]], spec))
        shuffle = getattr(c.train_loader, 'sampler', None)
        if hasattr(shuffle, 'generator'):
            shuffle.generator = torch.Generator().manual_seed(seed * 100003 + k + 1)
        local[c.id] = dict(gan=gan, loader=c.train_loader, shuffle=shuffle, slot=slots[k], size=sizes[k],
                           class_of_local={a: class_of[x] for a, x in enumerate(labels)})

    trainer, bb, history, start = GlobalClassifierTrainer(classifier_factory, config, device), BulletinBoard(), [], 0
    if ck:
        thetas, history, start = ck['thetas'], ck['history'], ck['round']
        if on_resume:
            on_resume(start)
        for cid, st in ck['clients'].items():
            local[cid]['gan'].load_state_dict(st)
            if st.get('shuffle') is not None:
                local[cid]['shuffle'].generator.set_state(st['shuffle'])
        if ck['trainer'] is not None:
            trainer.model = trainer.factory(len(existing)).to(device)
            trainer.model.load_state_dict(ck['trainer']['model'])
            trainer.optimizer = getattr(torch.optim, config.get('global_model_optim', 'Adam'))(
                trainer.model.parameters(), lr=config.get('global_model_optim_lr', 1e-3))
            trainer.optimizer.load_state_dict(ck['trainer']['optimizer'])
            trainer._mapping = ck['trainer']['mapping']
        _set_rng_state(ck['rng'])
    if workers > 1 and all(str(d) == 'cpu' for d in devices):
        import os
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))

    clip = config.get('upload_clip', 1e3)

    def train_client(c):
        v = local[c.id]
        n = v['gan'].train(v['loader'])
        return c.id, (v['gan'].update() if n else None)

    model = trainer.model
    bar = tqdm(range(start, rounds), desc='rounds', unit='round', initial=start, total=rounds, disable=not progress)
    for r in bar:
        row = dict(round=r + 1, seconds={}, bytes={})
        bar.set_postfix(stage='downlink')
        t = time.perf_counter()                                           # 4. G trunks broadcast
        if r:
            from secfl.broadcast import pack_state
            blobs = [pack_state(unflatten(th, spec)) for th in thetas]
            for s, blob in enumerate(blobs):
                bb.post('aggregator', f'trunk/{r}/{s}', hashlib.sha256(blob).digest())   # digest: no leak
            row['bytes']['broadcast'] = sum(map(len, blobs))
            for c in clients:
                local[c.id]['gan'].load_global(unflatten(thetas[local[c.id]['slot']], spec))
        row['seconds']['downlink'] = time.perf_counter() - t

        t = time.perf_counter()                                           # 5. local training
        bar.set_postfix(stage='local training')
        inner = tqdm(total=len(clients), desc=f'  round {r + 1} clients', unit='client', leave=False,
                     disable=not progress)
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
        updates = {cid: u for cid, u in results if u is not None}
        row['seconds']['local_training'] = time.perf_counter() - t

        bar.set_postfix(stage='secagg' if agg == 'secagg' else 'aggregation')
        t = time.perf_counter()                                           # 6. group means, no counts
        vectors = {cid: encode_upload(u, local[cid]['slot'], G, clip, 1 / local[cid]['size'])
                   for cid, u in updates.items()}
        if agg == 'secagg' and len(vectors) > 1:
            total, _ = run_secagg(vectors, threshold=max(2, -(-2 * len(vectors) // 3)), modulus_bits=64,
                                  session=f'cluster-round-{r}'.encode(), workers=workers)
        else:
            with np.errstate(over='ignore'):
                total = np.sum(list(vectors.values()), axis=0, dtype=np.uint64)
        thetas = thetas - decode_sum(total, G)                          # upload is -delta / |group|
        row['bytes']['upload'] = sum(v.size * 8 for v in vectors.values())
        row['seconds']['aggregation'] = time.perf_counter() - t

        bar.set_postfix(stage='global classifier')
        t = time.perf_counter()                                           # 7. global classifier
        gens, mapping = {}, {}
        for s, labels in enumerate(held):
            if not labels:
                continue
            g = gen_factory()
            load_generator_state(g, unflatten(thetas[s], spec))
            gens[str(s)] = GroupCoded(g.to(device).eval(), [code(x, code_dim) for x in labels])
            mapping[str(s)] = {a: class_of[x] for a, x in enumerate(labels)}
        model = trainer(gens, mapping)
        row['seconds']['global_training'] = time.perf_counter() - t

        bar.set_postfix(stage='evaluation')                              # 8. evaluation (evaluation.py)
        t = time.perf_counter()
        ev = evaluate_global(model, tests, predicted, truth, device)
        row['seconds']['evaluation'] = time.perf_counter() - t
        row['accuracy'] = ev['ground_truth_acc']
        row['old_acc'] = ev['old_acc']
        row['evaluation'] = ev
        row['union_metrics'] = union_metrics
        bar.set_postfix(acc=f"{row['accuracy']:.4f}")
        history.append(row)
        if record:
            record(row)
        if checkpoint_dir and ((r + 1) % save_every == 0 or r + 1 == rounds):
            t = time.perf_counter()
            state = dict(round=r + 1, thetas=thetas, cluster=cluster, agg=agg, code_dim=code_dim, history=history,
                         grouping=grouping, group_stats=group_stats, view=view, union=U,
                         clients={cid: dict(v['gan'].state_dict(), shuffle=getattr(getattr(v['shuffle'], 'generator', None),
                                                                                   'get_state', lambda: None)())
                                  for cid, v in local.items()},
                         trainer=dict(model=model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                                      mapping=trainer._mapping),
                         rng=_rng_state())
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(Path(checkpoint_dir) / 'checkpoint_last.pt', **state)
            if keep_all:
                save_checkpoint(Path(checkpoint_dir) / f'round_{r + 1:04d}.pt', **state)
            row['seconds']['checkpoint'] = time.perf_counter() - t
    bar.close()
    setup.update(trunk_params=int(thetas.shape[1]), resumed_from=start)
    return dict(history=history, setup=setup, model=model, thetas=thetas, grouping=grouping)


def main():
    from setup import parser as base_parser, seed_all, build_clients, resolve_device
    from omegaconf import OmegaConf
    p = base_parser()
    p.description = 'Secure pull protocol with private PACFL groups + per-group shared-trunk conditional GANs'
    p.add_argument('--cluster', dest='group_by', choices=('private', 'plain', 'original', 'none'), default='private')
    p.add_argument('--tau', type=float, default=.5, help='similarity threshold s_ij > tau links two clients')
    p.add_argument('--proj-dim', type=int, default=128, help='public random projection size r')
    p.add_argument('--agg', choices=('plain', 'secagg'), default='secagg')
    p.add_argument('--no-domain-check', action='store_true', help='oprf union: names only, no image check')
    p.add_argument('--samples-per-label', type=int, default=16, help='images per label for the domain check')
    p.add_argument('--union', choices=('oprf', 'mpc', 'secagg'), default='oprf',
                   help='label union: oprf = PSI-style n-party OPRF tags + SecAgg, no dictionary (default); '
                        'mpc = clients-only MPC over the dictionary; both give the Aggregator only the index '
                        'list. secagg = indicator vectors (Aggregator also learns names and holder counts)')
    p.add_argument('--code-dim', type=int, default=128)
    p.add_argument('--dictionary', type=Path, help='public dictionary: one canonical label id per line')
    p.add_argument('--devices', help='comma list, e.g. cuda:0,cuda:1 or mps,cpu; clients are spread round-robin')
    p.add_argument('--workers', type=int, default=1, help='parallel clients / preprocessing processes')
    p.add_argument('--save-every', type=int, default=1, help='checkpoint every k rounds (0: never)')
    p.add_argument('--keep-all', action='store_true', help='also keep round_XXXX.pt for every save')
    p.add_argument('--resume', type=Path, help='checkpoint to continue from (same experiment settings)')
    p.add_argument('--no-progress', action='store_true', help='no progress bars / group table')
    p.add_argument('--fast', action='store_true',
                   help='laptop test: 1 local GAN epoch, 1 classifier epoch, 32 samples/class, 3 rounds '
                        '(--rounds overrides), each client capped at --fast-samples train/test images')
    p.add_argument('--fast-samples', type=int, default=256, help='per-client image cap under --fast')
    args = p.parse_args()
    args.device = resolve_device(args.device)
    seed_all(args.seed)
    config = OmegaConf.to_container(OmegaConf.load(args.exp_conf), resolve=True)
    if args.fast:
        config.update(gen_local_epochs=1, global_model_epochs=1, global_samples_per_class=32)
        args.rounds = args.rounds or 3
    if args.smoke:
        from smoke import TinyClassifier
        config.update(gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4, gen_noise_dim=4)
        clients, spaces, tests = build_patterned(config)
        args.pacfl_basis_budget = 2                                  # 12-pixel images: 1 direction per label
        dictionary = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']
        code_dim = 8
        gen_f, disc_f, cls_f = (lambda: TinyCodeGenerator(code_dim, 4)), TinyLocalDiscriminator, TinyClassifier
    else:
        from nets import DCGANDiscriminator
        from secfl.code_gan import CodeDCGANGenerator
        clients, spaces, tests, _, _, cls_f = build_clients(args, config)
        from types import SimpleNamespace                              # drop the unused per-client
        clients = [SimpleNamespace(id=c.id, train_loader=c.train_loader) for c in clients]   # ResNet/GAN (~2 GB)
        if args.fast:
            for c in clients:
                c.train_loader = subsample(c.train_loader, args.fast_samples, args.seed + c.id, shuffle=True)
            tests = [(name, cid, subsample(l, args.fast_samples, args.seed + cid)) for name, cid, l in tests]
        dictionary = (args.dictionary.read_text().split('\n') if args.dictionary
                      else sorted({x for names in spaces.values() for x in names}))
        dictionary = [x for x in dictionary if x]
        code_dim, nd = args.code_dim, config.get('gen_noise_dim', 128)
        gen_f, disc_f = (lambda: CodeDCGANGenerator(code_dim, nd)), DCGANDiscriminator
    if args.resume:
        out = args.output or args.resume.parent
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = args.output or Path('runs') / (time.strftime('%Y%m%dT%H%M%S') + f'_cluster_{args.group_by}_{args.agg}')
        if (out / 'checkpoint_last.pt').exists():
            raise SystemExit(f'{out} already has a checkpoint: pass --resume {out / "checkpoint_last.pt"}')
        out.mkdir(parents=True, exist_ok=True)                            # a run that died before round 1
        trim_log(out / 'metrics.jsonl', 0)                                # leaves no stale rows behind
        (out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    def record(row):
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        from tqdm.auto import tqdm
        tqdm.write(f"Round {row['round']}: acc={row['accuracy']:.4f} old_acc={row['old_acc']:.4f} "
                   f"union MCC={row['union_metrics']['mcc']:.4f} upload={row['bytes']['upload']}B "
                   f"({sum(row['seconds'].values()):.1f}s)")
    devices = [resolve_device(d) for d in args.devices.split(',')] if args.devices else [args.device]
    result = run(clients, spaces, tests, gen_f, disc_f, cls_f, config, dictionary, cluster=args.group_by,
                 tau=args.tau, proj_dim=args.proj_dim, basis_budget=args.pacfl_basis_budget,
                 alpha=args.pacfl_cluster_alpha, agg=args.agg, rounds=args.rounds or config.get('global_rounds', 45),
                 code_dim=code_dim, device=devices[0], devices=devices, workers=args.workers, record=record,
                 checkpoint_dir=out if args.save_every else None, save_every=max(1, args.save_every),
                 keep_all=args.keep_all, resume=args.resume, seed=args.seed, progress=not args.no_progress,
                 on_resume=lambda r: trim_log(out / 'metrics.jsonl', r), union=args.union,
                 domain_check=not args.no_domain_check, samples_per_label=args.samples_per_label)
    write_json(out / 'setup.json', result['setup'])
    print(json.dumps({k: v for k, v in result['setup'].items() if k != 'experimenter_view'}, default=str))
    print(f'Artifacts: {out.resolve()}')


def build_patterned(config, label_sets=(('cat', 'dog'), ('cat', 'dog'), ('ship', 'truck'), ('ship', 'truck')), seed=0):
    """Tiny 2x2 RGB images; each label has a fixed random pattern, so PACFL has structure to find.
    Clients 0,1 and 2,3 share labels -> two groups."""
    from torch.utils.data import DataLoader, TensorDataset
    from types import SimpleNamespace
    g = torch.Generator().manual_seed(seed)
    names = sorted({x for s in label_sets for x in s})
    pattern = {x: torch.rand(3, 2, 2, generator=g) * 2 - 1 for x in names}
    clients, spaces, tests = [], {}, []
    for cid, labels in enumerate(label_sets):
        xs, ys = [], []
        for a, name in enumerate(labels):
            scale = torch.rand(8, 1, 1, 1, generator=g) * .5 + .5
            xs.append((pattern[name] * scale + .05 * torch.randn(8, 3, 2, 2, generator=g)).clamp(-1, 1))
            ys.append(torch.full((8,), a))
        x, y = torch.cat(xs), torch.cat(ys)
        clients.append(SimpleNamespace(id=cid, train_loader=DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)))
        spaces[cid] = labels
        tests.append(('synthetic', cid, DataLoader(TensorDataset(x, y), batch_size=8)))
    return clients, spaces, tests


if __name__ == '__main__':
    main()
