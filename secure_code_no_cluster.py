"""Secure pull protocol, NO-CLUSTER version: public label codes + ONE shared-trunk conditional GAN
for all clients (exact labels). The private-PACFL version clusters clients first.

Setup:
  1. clients name their classes by ids of a public dictionary (synonyms folded)
  2. code c_L = code_vector(SHA-256(L)) -- public, every holder of L gets the same code
  3. one SecAgg of dictionary indicator vectors (secfl.label_codes.discover): the Aggregator
     learns which labels exist and how many clients hold each, not who
Every round:
  4. Aggregator broadcasts the trunk G (same for everyone; initial G from a public seed)
  5. clients train G(z, c_L) on their own labels with a private local D
  6. -(theta_local - theta_global) and n go through SecAgg (--agg secagg) or in the clear
     (--agg plain); theta <- sample-weighted FedAvg (secfl.settle, lr=1)
  7. Aggregator trains the global classifier on G(z, c_L) for every delivered code
  8. evaluation on every client's test split (label -> code -> class)

  python -m secure_code_no_cluster --smoke --agg secagg --rounds 3
  python -m secure_code_no_cluster --devices cuda:0,cuda:1 --workers 4 --agg secagg --rounds 45
  python -m secure_code_no_cluster --resume runs/<run>/checkpoint_last.pt --rounds 60      # continue a run
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from secfl.bb import BulletinBoard, PublicParams
from secfl.code_gan import ClientCodeGAN, generator_state, load_generator_state, classifier_inputs
from label_union import discover, mpc_union, oprf_union, union_metrics as score_union, true_union, index_metrics
from secfl.label_codes import label_code, index_code
from secfl.settle import flatten, unflatten, settle
from secfl.upload import UploadLayout
from secure_main import aggregate
from evaluation import evaluate_global
from mapping import ByClassMapping
from training import GlobalClassifierTrainer

PUBLIC_SEED = 20260925


def seeded(factory, seed=PUBLIC_SEED):
    """Same initial weights on every machine, without touching the global RNG stream."""
    def build(*args):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return factory(*args)
    return build


CHECKPOINT_FORMAT = 'secure_code_main.v1'


def _rng_state():
    import random
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(state):
    import random
    torch.set_rng_state(state['torch']); np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, **state):
    """Atomic write: a crash mid-save never leaves a truncated checkpoint behind."""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(format=CHECKPOINT_FORMAT, **state), tmp)
    tmp.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)      # own files only
    if state.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{path}: not a {CHECKPOINT_FORMAT} checkpoint')
    return state


def union_relations(ids, names, class_of, keys=None):
    """(predicted, truth) relation tables for evaluation.py: predicted = the protocol's classes
    (client label -> its union key -> Aggregator class), truth = ByClassMapping over the clients'
    semantic names. keys: per client, the union key of each local label (default: the name)."""
    keys = keys or names
    predicted = {cid: {a: class_of[x] for a, x in enumerate(k)} for cid, k in zip(ids, keys)}
    return predicted, ByClassMapping(dict(zip(ids, names)))()


def label_samples(loader, names, k):
    """Up to k training images per local label (for the domain check), read in dataset order."""
    from torch.utils.data import DataLoader
    got = {}
    for x, y in DataLoader(loader.dataset, batch_size=256, shuffle=False):
        for xi, yi in zip(x, y.tolist()):
            if len(got.setdefault(yi, [])) < k:
                got[yi].append(xi.numpy())
        if len(got) == len(names) and all(len(v) >= k for v in got.values()):
            break
    return {names[a]: np.stack(v) for a, v in got.items()}


def build_union(names, dictionary, ids, method='mpc', workers=1, say=lambda *_: None, samples=None):
    """Label union, once before training. Returns a dict (stored in checkpoints, reused on resume):
      existing   the Aggregator's label table: names (secagg) or just indices 0..U-1 (oprf, mpc)
      keys       per client, the key of each local label in that table (name or index)
      metrics    label_union.union_metrics / index_metrics (experimenter only)
      view       experimenter-only plaintext table;  info: method stats
    Codes: secagg -> label_code(name); oprf / mpc -> index_code(index) (the Aggregator never sees names).
    The dictionary is only needed by secagg / mpc; oprf uses it just to score the union (experimenter)."""
    t = time.perf_counter()
    real, real_holders = true_union(names)                                # experimenter only
    if method == 'secagg':
        say(f'[setup] label union: SecAgg of {len(ids)} indicator vectors over a {len(dictionary)}-entry dictionary ...')
        existing, holders, info = discover(names, dictionary)
        keys = names
        metrics = score_union(existing, real, dictionary, dict(zip(existing, holders)), real_holders)
        view = [dict(cls=k, label=x, holders=c, clients=[i for i, l in zip(ids, names) if x in l])
                for k, (x, c) in enumerate(zip(existing, holders))]
        info = dict(info, holders=dict(zip(existing, holders)))
    elif method in ('mpc', 'oprf'):
        if method == 'oprf':
            domains = None
            if samples is not None:                                       # weak image check, client-local
                from label_union.domain import client_codes
                coded = [client_codes(s, l) for s, l in zip(samples, names)]
                domains = [d for d, _ in coded]
            say(f'[setup] label union: {len(ids)}-party OPRF tags + SecAgg (no dictionary'
                f'{", with domain check" if domains else ""}) ...')
            index, U, info = oprf_union(names, workers=workers, domains=domains)
            if domains:                                                   # experimenter only
                info['domains'] = {str(i): d for i, d in zip(ids, domains)}
                info['min_domain_margin'] = min(m for _, m in coded)
        else:
            say(f'[setup] label union: {len(ids)}-party MPC over a {len(dictionary)}-entry dictionary ...')
            index, U, info = mpc_union(names, dictionary, workers=workers)
        existing = list(range(U))
        keys = [[idx[x] for x in l] for idx, l in zip(index, names)]
        metrics = index_metrics(names, index, U, dictionary)
        view = [dict(cls=k, label='/'.join(sorted({x for l, idx in zip(names, index) for x in l if idx[x] == k})),
                     holders=sum(k in idx.values() for idx in index),
                     clients=[i for i, idx in zip(ids, index) if k in idx.values()]) for k in existing]
    else:
        raise ValueError('union must be oprf, mpc or secagg')
    info = dict(info, method=method, seconds=time.perf_counter() - t)
    say(f"[setup] label union done in {info['seconds']:.1f}s: Aggregator table has {len(existing)} entries; "
        f"MCC={metrics['mcc']:.4f} F1={metrics['f1']:.4f} exact={metrics['exact']} "
        f"missing={metrics['missing']} spurious={metrics['spurious']}")
    return dict(existing=existing, keys=keys, metrics=metrics, view=view, info=info)


def write_json(path, obj):
    """Atomic JSON write (a crash never leaves a half file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def trim_log(path, rounds):
    """On resume: drop rows after the checkpoint's round, so reruns are not logged twice. A line
    a crash left half-written is dropped too (it is always after the checkpoint). Atomic rewrite."""
    path = Path(path)
    if path.exists():
        keep = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row['round'] <= rounds:
                keep.append(line)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(''.join(r + '\n' for r in keep))
        tmp.replace(path)


def format_union(rows, max_rows=200):
    lines = ['[experimenter view: plaintext union, not visible to any party; the Aggregator learns only '
             'the class column (mpc), or class/label/holders (secagg)]',
             f"{'class':>5}  {'label':<24} {'holders':>7}  clients"]
    for r in rows[:max_rows]:
        lines.append(f"{r['cls']:>5}  {str(r['label'])[:24]:<24} {r['holders']:>7}  {r['clients']}")
    if len(rows) > max_rows:
        lines.append(f'... {len(rows) - max_rows} more (see union_plain.json)')
    return '\n'.join(lines)


def run(clients, label_spaces, tests, gen_factory, disc_factory, classifier_factory, config, dictionary,
        agg='secagg', rounds=3, code_dim=128, device='cpu', record=None, devices=None, workers=1,
        checkpoint_dir=None, save_every=1, keep_all=False, resume=None, seed=0, progress=False, on_resume=None,
        union='oprf', domain_check=True, samples_per_label=16):
    """gen_factory() -> trunk taking (z, codes); disc_factory(k) -> local D over k labels.

    label_spaces {client: ordered dictionary ids of its local labels}; dictionary: public ids.
    devices: list of devices; client i trains on devices[i % len]. workers: clients trained
    concurrently (threads) and SecAgg masking threads. checkpoint_dir/save_every/keep_all:
    full state after every save_every rounds (checkpoint_last.pt, plus round_XXXX.pt if
    keep_all). progress: tqdm bars on stderr. resume: checkpoint path; continues after its last completed round up to
    `rounds` total, bit-for-bit identical to an uninterrupted run. Every client has private
    noise and shuffle RNGs, so results do not depend on workers either.
    """
    if agg not in ('plain', 'secagg'):
        raise ValueError('agg must be plain or secagg')
    devices = list(devices or [device])
    ids = [c.id for c in clients]
    names = [list(label_spaces[i]) for i in ids]
    from tqdm.auto import tqdm
    say = tqdm.write if progress else (lambda *_: None)
    ck = load_checkpoint(resume) if resume else None
    if ck and (ck.get('union', {}).get('info', {}).get('method') != union or ck['agg'] != agg
               or ck['code_dim'] != code_dim or ck['clients'].keys() != set(ids)):
        raise ValueError('checkpoint does not match this experiment (union, agg, code_dim or clients)')
    samples = ([label_samples(c.train_loader, n, samples_per_label) for c, n in zip(clients, names)]
               if union == 'oprf' and domain_check and not ck else None)
    U = ck['union'] if ck else build_union(names, dictionary, ids, union, workers, say, samples)   # one-time
    existing, keys, union_metrics = U['existing'], U['keys'], U['metrics']
    if progress:
        say(format_union(U['view']))
    code = label_code if union == 'secagg' else index_code
    class_of = {x: k for k, x in enumerate(existing)}                      # Aggregator's class ids
    agg_codes = [code(x, code_dim) for x in existing]
    predicted, truth = union_relations(ids, names, class_of, keys)       # for global accuracy
    setup = dict(union=U['info'], labels=len(existing), union_metrics=union_metrics, experimenter_view=U['view'])
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(checkpoint_dir) / 'setup.json', setup)          # early: a crashed run still has it

    gen_factory = seeded(gen_factory)
    theta, spec = flatten(generator_state(gen_factory()))
    local = {}
    for k, (c, labels) in enumerate(zip(clients, keys)):
        dev = devices[k % len(devices)]
        codes = {a: code(x, code_dim) for a, x in enumerate(labels)}
        gan = ClientCodeGAN(codes, gen_factory(), disc_factory(len(labels)), config, dev, seed=seed * 100003 + k)
        gan.load_global(unflatten(theta, spec))
        shuffle = getattr(c.train_loader, 'sampler', None)
        if hasattr(shuffle, 'generator'):                                # private shuffle RNG per client:
            shuffle.generator = torch.Generator().manual_seed(seed * 100003 + k + 1)   # thread-order independent
        local[c.id] = dict(gan=gan, loader=c.train_loader, shuffle=shuffle,
                           class_of_local={a: class_of[x] for a, x in enumerate(labels)})

    layout = None
    if agg == 'secagg':
        n_max = sum(len(c.train_loader.dataset) for c in clients)
        params = PublicParams(labels=('G',), modulus_bits=64, frac_bits=24,
                              clip=config.get('upload_clip', 1e3), threshold=1)
        layout = UploadLayout(params, {'G': theta.size}, n_max, len(clients))
    trainer, bb, history = GlobalClassifierTrainer(classifier_factory, config, device), BulletinBoard(), []
    start = 0
    if ck:
        theta, history, start = ck['theta'], ck['history'], ck['round']
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
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))   # avoid oversubscription

    def train_client(c):
        n = local[c.id]['gan'].train(local[c.id]['loader'])
        return c.id, (({'G': local[c.id]['gan'].update()}, {'G': n}) if n else None)

    model = trainer.model
    bar = tqdm(range(start, rounds), desc='rounds', unit='round', initial=start, total=rounds, disable=not progress)
    for r in bar:
        row = dict(round=r + 1, seconds={}, bytes={})
        bar.set_postfix(stage='downlink')
        t = time.perf_counter()                                           # 4. trunk broadcast
        if r:
            from secfl.broadcast import pack_state
            blob = pack_state(unflatten(theta, spec))
            bb.post('aggregator', f'trunk/{r}', hashlib.sha256(blob).digest())   # digest only: the full
            # blob every round grew memory by |trunk| per round (GBs over a long run)
            row['bytes']['broadcast'] = len(blob)
            for c in clients:
                local[c.id]['gan'].load_global(unflatten(theta, spec))
        row['seconds']['downlink'] = time.perf_counter() - t

        t = time.perf_counter()                                           # 5. local training
        bar.set_postfix(stage='local training')
        inner = tqdm(total=len(clients), desc=f'  round {r + 1} clients', unit='client', leave=False,
                     disable=not progress)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(train_client, c) for c in clients]
                for f in as_completed(futures):
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
        t = time.perf_counter()                                           # 6. aggregation
        grad_sums, N, G, row['bytes']['upload'] = aggregate(updates, layout, f'tag-round-{r}'.encode(), workers)
        new, _ = settle({'G': theta}, grad_sums, N, G, lr=1.0, threshold=1)
        theta = new['G']
        row['seconds']['aggregation'] = time.perf_counter() - t

        bar.set_postfix(stage='global classifier')
        t = time.perf_counter()                                           # 7. global classifier
        g = gen_factory()
        load_generator_state(g, unflatten(theta, spec))
        model = trainer(*classifier_inputs(g.to(device).eval(), agg_codes))
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
            state = dict(round=r + 1, theta=theta, existing=existing, union=U, agg=agg, code_dim=code_dim,
                         history=history,
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
    setup.update(trunk_params=int(theta.size), resumed_from=start)
    return dict(history=history, setup=setup, model=model, theta=theta)


# ---------------------------------------------------------------------------- smoke models
class TinyCodeGenerator(nn.Module):
    def __init__(self, code_dim=8, noise_dim=4):
        super().__init__()
        self.linear = nn.Linear(noise_dim + code_dim, 12)

    def forward(self, z, codes):
        return self.linear(torch.cat([z, codes], 1)).tanh().reshape(-1, 3, 2, 2)


class TinyLocalDiscriminator(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.embedding, self.linear = nn.Embedding(k, 2), nn.Linear(14, 1)

    def forward(self, images, labels):
        return self.linear(torch.cat([images.flatten(1), self.embedding(labels)], 1))


def subsample(loader, n, seed, shuffle=False):
    """Same loader over a fixed random subset of at most n items (--fast)."""
    from torch.utils.data import DataLoader, Subset
    ds = loader.dataset
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=loader.batch_size, shuffle=shuffle)


def main():
    from setup import parser as base_parser, seed_all, build_clients
    from omegaconf import OmegaConf
    p = base_parser()
    p.description = 'Secure pull protocol with public label codes + shared-trunk conditional GAN (exact)'
    p.add_argument('--agg', choices=('plain', 'secagg'), default='secagg',
                   help='secagg: the protocol; plain: no cryptography (with --gen cbn: Plain-GeFL baseline, '
                        'plaintext union, rows handed out directly, same model and update rule)')
    p.add_argument('--gen', choices=('cbn', 'code'), default='cbn',
                   help='cbn: shared trunk + per-label conditional-BatchNorm rows, rows sent by KEM so only '
                        'holders get them (secure_cbn, needs --union oprf); code: fixed public label codes')
    p.add_argument('--no-domain-check', action='store_true', help='oprf union: names only, no image check')
    p.add_argument('--samples-per-label', type=int, default=16, help='images per label for the domain check')
    p.add_argument('--union', choices=('oprf', 'fuzzy', 'mpc', 'secagg'), default='oprf',
                   help='label union: oprf = PSI-style n-party OPRF tags + SecAgg on the label names (default); '
                        'fuzzy = no shared names: every client names its labels with its own keyword in its own language '
                        '(--fuzzy-langs), snapped locally to public anchor classes by a cross-lingual encoder, then exact union (label_union.fuzzy_union; '
                        'parameters from tests/fuzzy_threshold.py); mpc = clients-only MPC over the dictionary; '
                        'secagg = indicator vectors (Aggregator also learns names and holder counts)')
    p.add_argument('--fuzzy-langs', default='en0,en1,en2',
                   help='--union fuzzy: client i writes its label keywords as writer i mod len (en0,en1,en2 = English, '
                        'different wordings; or en,zh,es,ja,fr,de; rt_descriptions.keyword)')
    p.add_argument('--code-dim', type=int, default=128)
    p.add_argument('--dictionary', type=Path, help='public dictionary: one canonical label id per line')
    p.add_argument('--devices', help='comma list, e.g. cuda:0,cuda:1 or mps,cpu; clients are spread round-robin')
    p.add_argument('--workers', type=int, default=1, help='clients trained in parallel (threads)')
    p.add_argument('--save-every', type=int, default=1, help='checkpoint every k rounds (0: never)')
    p.add_argument('--keep-all', action='store_true', help='also keep round_XXXX.pt for every save')
    p.add_argument('--resume', type=Path, help='checkpoint to continue from (same experiment settings)')
    p.add_argument('--no-progress', action='store_true', help='no progress bars / union table')
    p.add_argument('--fast', action='store_true',
                   help='laptop test: 1 local GAN epoch, 1 classifier epoch, 32 samples/class, 3 rounds '
                        '(--rounds overrides), each client capped at --fast-samples train/test images')
    p.add_argument('--no-quantize', action='store_true', help='cbn + secagg: 64-bit fixed point, no compression')
    p.add_argument('--warmup-epochs', type=int, default=0,
                   help='cbn: local generator epochs per client before round 1 (nothing uploaded)')
    p.add_argument('--min-holders', type=int, default=2,
                   help='cbn: a label row is updated only if at least this many clients contributed')
    p.add_argument('--keep-frac', type=float, default=0.1, help='cbn + secagg: coordinates uploaded per round')
    p.add_argument('--quant-scale0', type=float, default=0.05, help='cbn + secagg: first-round scale floor')
    p.add_argument('--fast-samples', type=int, default=256, help='per-client image cap under --fast')
    args = p.parse_args()
    from setup import resolve_device
    args.device = resolve_device(args.device)
    seed_all(args.seed)
    config = OmegaConf.to_container(OmegaConf.load(args.exp_conf), resolve=True)
    if args.fast:
        config.update(gen_local_epochs=1, global_model_epochs=1, global_samples_per_class=32)
        args.rounds = args.rounds or 3
    if args.smoke:
        from secure_main import build_synthetic
        from smoke import TinyClassifier
        config.update(gen_local_epochs=1, global_model_epochs=1, batch_size=4, global_samples_per_class=4, gen_noise_dim=4)
        clients, spaces, tests = build_synthetic(config)
        dictionary = ['bird', 'cat', 'dog', 'horse', 'ship', 'truck']
        code_dim = 8
        gen_f, disc_f, cls_f = (lambda: TinyCodeGenerator(code_dim, 4)), TinyLocalDiscriminator, TinyClassifier
        if args.gen == 'cbn':
            from secfl.cbn_gan import TinyCBNGenerator
            gen_f = lambda k: TinyCBNGenerator(k, 4, 2)
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
        # ponytail: default dictionary = the public class lists of the loaded datasets
        dictionary = (args.dictionary.read_text().split('\n') if args.dictionary
                      else sorted({x for names in spaces.values() for x in names}))
        dictionary = [x for x in dictionary if x]
        code_dim, nd = args.code_dim, config.get('gen_noise_dim', 128)
        gen_f, disc_f = (lambda: CodeDCGANGenerator(code_dim, nd)), DCGANDiscriminator
        if args.gen == 'cbn':
            from secfl.cbn_gan import CBNGenerator
            gen_f = lambda k: CBNGenerator(k, nd)
    if args.gen == 'cbn' and args.union not in ('oprf', 'fuzzy'):
        raise SystemExit('--gen cbn needs --union oprf or fuzzy (the KEM keys come from the union)')
    if args.union == 'fuzzy' and args.gen != 'cbn':
        raise SystemExit('--union fuzzy needs --gen cbn')
    keywords = fuzzy = None
    if args.union == 'fuzzy':                                             # each client's own words
        from rt_descriptions import keyword
        langs = args.fuzzy_langs.split(',')
        dataset_of = {cid: name for name, cid, _ in tests}
        keywords = [{x: keyword(dataset_of.get(c.id, ''), x, langs[j % len(langs)]) for x in spaces[c.id]}
                    for j, c in enumerate(clients)]
        fuzzy = dict(langs=[langs[j % len(langs)] for j in range(len(clients))])   # each client knows its own
    if args.resume:
        out = args.output or args.resume.parent                            # keep appending to that run
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = args.output or Path('runs') / (time.strftime('%Y%m%dT%H%M%S') + f'_no_cluster_{args.gen}_{args.agg}')
        if (out / 'checkpoint_last.pt').exists():
            raise SystemExit(f'{out} already has a checkpoint: pass --resume {out / "checkpoint_last.pt"}')
        out.mkdir(parents=True, exist_ok=True)                            # a run that died before round 1
        trim_log(out / 'metrics.jsonl', 0)                                # leaves no stale rows behind
        (out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    def record(row):
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        from tqdm.auto import tqdm
        um = row.get('union_metrics')
        per = row['bytes'].get('upload_per_client', row['bytes']['upload'])
        tqdm.write(f"Round {row['round']}: acc={row['accuracy']:.4f} old_acc={row['old_acc']:.4f} "
                   + (f"union exact={um['exact']} " if um else '')
                   + f"upload/client={per / 1e3:.1f}kB ({sum(row['seconds'].values()):.1f}s)")
    devices = [resolve_device(d) for d in args.devices.split(',')] if args.devices else [args.device]
    common = dict(agg=args.agg, rounds=args.rounds or config.get('global_rounds', 45),
                  device=devices[0], devices=devices, workers=args.workers, record=record,
                  checkpoint_dir=out if args.save_every else None, save_every=max(1, args.save_every),
                  keep_all=args.keep_all, resume=args.resume, seed=args.seed, progress=not args.no_progress,
                  on_resume=lambda r: trim_log(out / 'metrics.jsonl', r),
                  domain_check=not args.no_domain_check, samples_per_label=args.samples_per_label)
    if args.gen == 'cbn':
        import secure_cbn
        result = secure_cbn.run(clients, spaces, tests, gen_f, disc_f, cls_f, config, dictionary,
                                quantize=not args.no_quantize, keep_frac=args.keep_frac,
                                quant_scale0=args.quant_scale0, min_holders=args.min_holders,
                                warmup_epochs=args.warmup_epochs,
                                union='fuzzy' if args.union == 'fuzzy' else 'exact', keywords=keywords, fuzzy=fuzzy, **common)
    else:
        result = run(clients, spaces, tests, gen_f, disc_f, cls_f, config, dictionary, code_dim=code_dim,
                     union=args.union, **common)
    ev = result.get('evaluator') or dict(union_metrics=result['setup'].pop('union_metrics', None),
                                         experimenter_view=result['setup'].pop('experimenter_view', None))
    write_json(out / 'evaluator' / 'union_plain.json', ev['experimenter_view'])   # experimenter only
    write_json(out / 'evaluator' / 'union_metrics.json', ev['union_metrics'])
    write_json(out / 'setup.json', result['setup'])
    print(json.dumps(result['setup'], default=str))
    print(f"union exact={ev['union_metrics']['exact']} (details: evaluator/)")
    print(f'Artifacts: {out.resolve()}')


if __name__ == '__main__':
    main()
