"""Experimenter-only diagnostics per round (--diagnostics): never part of the protocol, uses real data
and ground truth that no party has. Answers "why is accuracy what it is":

  ceiling      a reference classifier (same architecture as the global one) trained centrally on ALL
               clients' real training images: the accuracy synthetic data could at best reach
  per_class    global classifier test accuracy per (dataset, true class), and its top confusions
  fidelity     per generator row: share of its images the reference classifier calls its class
               (low = the generator does not draw that class)
  synth_fit    per row: share of its images the global classifier gets right (low = classifier
               underfits the synthetic data; high fit + low test accuracy = synthetic/real gap)
  diversity    per row: mean pixel std over its samples (near 0 = mode collapse)
  grid         <out>/diag/round_XXXX.png: 8 samples per row, rows in class order
"""
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


class _OwnRandomness:
    """Diagnostics must not change the run: save and restore the global CPU / CUDA random states."""
    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


def _logits(model, x):
    out = model(x)
    return out[1] if isinstance(out, tuple) else out


class Diagnostics:
    def __init__(self, *args, **kw):
        with _OwnRandomness():
            self._init(*args, **kw)

    def _init(self, train_loaders, ids, truth, predicted, target_names, classifier_factory, config, device,
              out_dir, view=None):
        from evaluation import semantic_alignment
        self.truth, self.names, self.device, self.out = truth, target_names, device, Path(out_dir)
        self.align = semantic_alignment(predicted, truth)                # generator row -> true class
        self.row_name = {k: f"{k}:{(view or {}).get(k, {}).get('label', target_names[t])}"
                         for k, t in self.align.items()}
        self.n = int(config.get('diag_samples', 200))
        self.out.mkdir(parents=True, exist_ok=True)
        self.ref = classifier_factory(len(target_names)).to(device)
        f = self.out / 'reference_pooled.pt'       # (reference.pt: older runs, trained client by client)
        if f.exists():
            self.ref.load_state_dict(torch.load(f, map_location=device))
        else:                                      # centralised: all clients' real images POOLED and shuffled
            xs, ts = [], []                        # (client after client, the last clients' classes win)
            for cid, loader in zip(ids, train_loaders):
                t = torch.tensor([truth[cid][i] for i in range(len(truth[cid]))])
                batches = (loader.ordered() if hasattr(loader, 'ordered') else   # dataset order: not the
                           torch.utils.data.DataLoader(loader.dataset, batch_size=256))  # client's shuffle RNG
                for x, y in batches:               # stored as uint8: the images are 8-bit anyway
                    xs.append(torch.round((x * .5 + .5) * 255).clamp_(0, 255).to(torch.uint8))
                    ts.append(t[y])
            X, T = torch.cat(xs), torch.cat(ts)
            opt = torch.optim.Adam(self.ref.parameters(), lr=config.get('global_model_optim_lr', 1e-3))
            bs, gen = int(config.get('batch_size', 64)), torch.Generator().manual_seed(0)
            self.ref.train()
            for _ in range(int(config.get('diag_ref_epochs', 10))):
                perm = torch.randperm(len(T), generator=gen)
                for s in range(0, len(T), bs):
                    j = perm[s:s + bs]
                    x = X[j].to(device).float().div_(255).sub_(.5).div_(.5)
                    opt.zero_grad()
                    F.cross_entropy(_logits(self.ref, x), T[j].to(device)).backward()
                    opt.step()
            torch.save(self.ref.state_dict(), f)
        self.ref.eval()
        self.ceiling = None

    def _test(self, model, tests, pred_name):
        hit, n, conf = Counter(), Counter(), Counter()
        with torch.no_grad():
            for dataset, group, loader in tests:
                for x, y in loader:
                    p = _logits(model, x.to(self.device)).argmax(1).cpu().tolist()
                    for pi, yi in zip(p, y.tolist()):
                        true = self.names[self.truth[group][int(yi)]]
                        key = f'{dataset}:{true}'
                        pred = pred_name(pi)
                        n[key] += 1
                        hit[key] += int(pred == true)
                        if pred != true:
                            conf[(key, pred)] += 1
        return hit, n, conf

    def __call__(self, *args):
        with _OwnRandomness():
            return self._round(*args)

    def _round(self, rnd, model, generator, U, tests):
        model.eval()
        if self.ceiling is None:                                         # reference on real test data
            hit, n, _ = self._test(self.ref, tests, lambda p: self.names[p])
            self.ceiling = dict(acc=sum(hit.values()) / max(1, sum(n.values())),
                                per_class={k: round(hit[k] / n[k], 4) for k in sorted(n)})
        hit, n, conf = self._test(model, tests, lambda p: self.names[self.align[p]] if p in self.align
                                  else 'merged/unaligned')
        fid, fit, div, grid = {}, {}, {}, []
        g = generator.to(self.device).eval()
        with torch.no_grad():
            for k in range(U):
                x = g(torch.randn(self.n, getattr(g, 'noise_dim', 128), device=self.device),
                      torch.full((self.n,), k, device=self.device))
                name = self.row_name.get(k, f'{k}:merged')
                if k in self.align:
                    fid[name] = round((_logits(self.ref, x).argmax(1) == self.align[k]).float().mean().item(), 4)
                fit[name] = round((_logits(model, x).argmax(1) == k).float().mean().item(), 4)
                div[name] = round(x.std(0).mean().item(), 4)
                grid.append(x[:8].cpu())
        try:
            from torchvision.utils import save_image
            save_image(torch.cat(grid), self.out / f'round_{rnd:04d}.png', nrow=8, normalize=True,
                       value_range=(-1, 1))
            (self.out / 'rows.json').write_text(json.dumps([self.row_name.get(k, f'{k}:merged') for k in range(U)]))
        except Exception as e:                                           # a picture must not stop a run
            print(f'[diagnostics] grid not saved: {e}', flush=True)
        mean = lambda d: round(sum(d.values()) / len(d), 4) if d else None
        return dict(ceiling=round(self.ceiling['acc'], 4), ceiling_per_class=self.ceiling['per_class'],
                    per_class={k: round(hit[k] / n[k], 4) for k in sorted(n)},
                    confusions=[[t, p, c] for (t, p), c in conf.most_common(15)],
                    fidelity=fid, fidelity_mean=mean(fid), synth_fit=fit, synth_fit_mean=mean(fit),
                    diversity=div, diversity_mean=mean(div))
