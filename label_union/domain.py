"""Weak image check for the label union: which KIND of picture a label is (strokes on a plain
background vs natural photo ...), not what is in it. Used as part of the PSI input:
tag = OPRF(name | domain code), so a digit "cat" and a photo "cat" get different indices while
MNIST "3" and EMNIST "3" still match.

signature(x)   per-image statistics that do not depend on content, rotation or flips:
               brightness histogram, colourfulness, edge strength, radial power spectrum.
anchors()      signatures of SYNTHETIC images made by public code + a public seed (no dataset,
               no download): every client computes the same anchors locally.
domain_code(x) nearest anchor of the label's mean signature; margin = how clear the decision was.
Everything stays on the client; only the code (one short word) enters the OPRF.
"""
import functools

import numpy as np

ANCHOR_SEED = 2026
ANCHOR_SAMPLES = 64


def signature(x, lo=-1.0) -> np.ndarray:
    """x: K x C x H x W images of one label with values in [lo, 1] (the pipelines use [-1, 1])
    -> one signature vector (Hellinger scale)."""
    x = (np.asarray(x, np.float64) - lo) / (1 - lo)
    if x.ndim == 3:
        x = x[None]
    if x.shape[1] == 1:
        x = np.repeat(x, 3, 1)
    g = x.mean(1)
    n, h, w = g.shape
    hist = lambda a, bins: np.stack([np.histogram(i, bins, (0, 1))[0] for i in a]) / (h * w)
    feats = [hist(g, 16), hist(x.max(1) - x.min(1), 8)]                 # brightness, colourfulness
    gy, gx = np.gradient(g, axis=(1, 2))
    feats.append(hist(np.clip(np.hypot(gx, gy), 0, 1), 8))              # edge strength
    P = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean((1, 2), keepdims=True)), axes=(1, 2))) ** 2
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    ring = np.minimum((np.hypot(yy / (h / 2), xx / (w / 2)) * 8).astype(int), 7).ravel()
    spec = np.log1p(np.stack([np.bincount(ring, p.ravel(), 8) for p in P]))
    feats.append(spec / np.maximum(spec.sum(1, keepdims=True), 1e-12))  # radial spectrum
    return np.sqrt(np.concatenate(feats, 1).mean(0))


def _strokes(k, g, size):
    from PIL import Image, ImageDraw
    out = []
    for _ in range(k):
        im = Image.new('L', (size, size), 0)
        d = ImageDraw.Draw(im)
        for _ in range(g.integers(1, 4)):
            pts = [tuple(int(v) for v in g.integers(size // 5, size - size // 5, 2)) for _ in range(g.integers(2, 5))]
            d.line(pts, fill=255, width=int(g.integers(max(1, size // 16), max(2, size // 8))))
        out.append(np.repeat((np.asarray(im, float) / 255)[None], 3, 0))
    return np.stack(out)


def _photo(k, g, size):
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing='ij')
    f = np.hypot(fy, fx)
    f[0, 0] = 1
    pink = lambda: np.real(np.fft.ifft2(np.fft.fft2(g.standard_normal((size, size))) / f))    # 1/f noise
    out = []
    for _ in range(k):
        base = pink()
        x = np.stack([base + .5 * pink() for _ in range(3)])
        out.append((x - x.min()) / (x.max() - x.min()))
    return np.stack(out)


GENERATORS = {'strokes': _strokes, 'photo': _photo}                     # add a domain = add a generator


def anchors(size=32, seed=ANCHOR_SEED):
    g = np.random.default_rng(seed)
    return {name: signature(gen(ANCHOR_SAMPLES, g, size), lo=0.) for name, gen in GENERATORS.items()}


def domain_code(x, anchor_sigs=None):
    """-> (code, margin). margin = cosine gap between the best and second-best anchor."""
    x = np.asarray(x)
    anchor_sigs = anchor_sigs or anchors(x.shape[-1])
    s = signature(x)
    cos = {k: float(s @ a / np.linalg.norm(s) / np.linalg.norm(a)) for k, a in anchor_sigs.items()}
    best = sorted(cos, key=cos.get, reverse=True)
    return best[0], cos[best[0]] - cos[best[1]]


def client_codes(samples, labels):
    """Domain code of every label a client declares. samples {label: images} may miss labels the
    client has (almost) no images of (e.g. a Dirichlet split); those get the code of all of the
    client's sampled images together (<= samples_per_label per label; same dataset, same kind of
    picture). -> ({label: code}, min margin)."""
    if not samples:
        raise ValueError('domain check needs at least one image')
    A = anchors(next(iter(samples.values())).shape[-1])
    coded = {x: domain_code(v, A) for x, v in samples.items()}
    missing = [x for x in labels if x not in coded]
    if missing:
        fallback = domain_code(np.concatenate(list(samples.values())), A)
        coded.update({x: fallback for x in missing})
    return {x: coded[x][0] for x in labels}, min(m for _, m in coded.values())


# ------------------------------------------------------------------ t-out-of-k image matching
# Fine-grained version of the check above, used by label_union.oprf_union.pairs_union: many public
# anchors, and a label is described by its k NEAREST anchors instead of one code. Two labels with the
# same keyword match when their k-sets share >= t anchors (t-out-of-k fuzzy matching). Every anchor is
# the signature of synthetic pictures from public code + a public seed: every client builds the same set.

def _px(size, frac):
    return max(1, int(round(size * frac)))


def _draw(k, g, size, mode, background, paint):
    from PIL import Image, ImageDraw
    out = []
    for _ in range(k):
        bg = background(g)
        im = Image.new(mode, (size, size), bg)
        paint(ImageDraw.Draw(im), g)
        a = np.asarray(im, float) / 255
        out.append(np.repeat(a[None], 3, 0) if a.ndim == 2 else a.transpose(2, 0, 1))
    return np.stack(out)


def _rgb(g):
    return tuple(int(v) for v in g.integers(0, 256, 3))


def _lines(k, g, size, width, count, invert=False, closed=False):
    fg, w = (0 if invert else 255), _px(size, width)

    def paint(d, g):
        for _ in range(int(g.integers(*count))):
            if closed:
                x0, y0 = (int(v) for v in g.integers(0, max(1, size // 2), 2))
                bw, bh = (int(v) for v in g.integers(max(1, size // 4), max(2, size // 2), 2))
                (d.ellipse if g.random() < .5 else d.rectangle)([x0, y0, x0 + bw, y0 + bh], outline=fg, width=w)
            else:
                pts = [tuple(int(v) for v in g.integers(size // 5, size - size // 5, 2))
                       for _ in range(int(g.integers(2, 5)))]
                d.line(pts, fill=fg, width=w)
    return _draw(k, g, size, 'L', lambda g: 255 - fg, paint)


def _shapes(k, g, size, radius, count, color, rect=False):
    r = _px(size, radius)

    def paint(d, g):
        for _ in range(int(g.integers(*count))):
            x, y = (int(v) for v in g.integers(0, size, 2))
            (d.rectangle if rect else d.ellipse)([x - r, y - r, x + r, y + r], fill=_rgb(g) if color else 255)
    return _draw(k, g, size, 'RGB' if color else 'L', _rgb if color else (lambda g: 0), paint)


def _scale(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def _pink(k, g, size, alpha, color):
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing='ij')
    f = np.hypot(fy, fx)
    f[0, 0] = 1
    field = lambda: np.real(np.fft.ifft2(np.fft.fft2(g.standard_normal((size, size))) / f ** alpha))
    out = []
    for _ in range(k):
        base = field()
        out.append(_scale(np.stack([base + .5 * field() for _ in range(3)]) if color else np.repeat(base[None], 3, 0)))
    return np.stack(out)


def _gradient(k, g, size, color):
    yy, xx = np.mgrid[0:size, 0:size] / size
    out = []
    for _ in range(k):
        th = g.uniform(0, 2 * np.pi)
        ramp = np.cos(th) * xx + np.sin(th) * yy
        out.append(_scale(np.stack([ramp * g.uniform(.2, 1) + g.uniform(0, .5) for _ in range(3)]) if color
                          else np.repeat(ramp[None], 3, 0)))
    return np.stack(out)


def _stripes(k, g, size, freq, color, square=False):
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for _ in range(k):
        th = g.uniform(0, np.pi)
        w = np.sin(2 * np.pi * freq * g.uniform(.8, 1.25) * (np.cos(th) * xx + np.sin(th) * yy) / size
                   + g.uniform(0, 2 * np.pi))
        w = np.sign(w) if square else w
        out.append(_scale(np.stack([w * g.uniform(.5, 1) for _ in range(3)]) if color else np.repeat(w[None], 3, 0)))
    return np.stack(out)


def _white(k, g, size, color, gauss):
    draw = g.standard_normal if gauss else g.random
    return np.stack([_scale(draw((3, size, size)) if color else np.repeat(draw((1, size, size)), 3, 0))
                     for _ in range(k)])


ANCHOR_FAMILIES = (                     # 45 anchors: (name, generator, parameters)
    [(f'strokes/w{w:.2f}/n{c[0]}', _lines, dict(width=w, count=c)) for w in (1 / 16, 1 / 10, 1 / 7) for c in ((1, 3), (3, 6))]
    + [(f'ink/w{w:.2f}/n{c[0]}', _lines, dict(width=w, count=c, invert=True)) for w in (1 / 16, 1 / 8) for c in ((1, 3), (3, 6))]
    + [(f'outline/w{w:.2f}/i{int(i)}', _lines, dict(width=w, count=(1, 4), closed=True, invert=i))
       for w in (1 / 16, 1 / 8) for i in (False, True)]
    + [(f'dots/r{r}/n{c[0]}', _shapes, dict(radius=r, count=c, color=False)) for r in (.05, .12) for c in ((3, 8), (10, 25))]
    + [(f'dots-color/r{r}', _shapes, dict(radius=r, count=(3, 8), color=True)) for r in (.08, .2)]
    + [(f'blocks/n{c[0]}', _shapes, dict(radius=.15, count=c, color=True, rect=True)) for c in ((2, 5), (6, 15))]
    + [(f'photo/a{a}', _pink, dict(alpha=a, color=True)) for a in (.6, .9, 1.2, 1.5, 2., 2.5)]
    + [(f'photo-gray/a{a}', _pink, dict(alpha=a, color=False)) for a in (.9, 1.5, 2.2)]
    + [(f'gradient/c{int(c)}', _gradient, dict(color=c)) for c in (False, True)]
    + [(f'stripes/f{f}/c{int(c)}', _stripes, dict(freq=f, color=c)) for f in (2, 5, 10) for c in (False, True)]
    + [(f'checker/f{f}', _stripes, dict(freq=f, color=False, square=True)) for f in (3, 8)]
    + [(f'noise/c{int(c)}/g{int(gs)}', _white, dict(color=c, gauss=gs)) for c in (False, True) for gs in (False, True)])


@functools.lru_cache(maxsize=8)
def anchor_matrix(size=32, seed=ANCHOR_SEED):
    """(names, unit-norm signatures N x d) of the public anchors at this image size."""
    g = np.random.default_rng(seed)
    S = np.stack([signature(gen(ANCHOR_SAMPLES, g, size, **kw), lo=0.) for _, gen, kw in ANCHOR_FAMILIES])
    return tuple(n for n, _, _ in ANCHOR_FAMILIES), S / np.linalg.norm(S, axis=1, keepdims=True)


def anchor_set(x, k=6, A=None):
    """Indices of the k anchors nearest (cosine) to the mean signature of x (images of one label)."""
    x = np.asarray(x)
    S = anchor_matrix(x.shape[-1])[1] if A is None else A
    s = signature(x)
    return tuple(sorted(int(j) for j in np.argsort(-(S @ (s / np.linalg.norm(s))), kind='stable')[:k]))


def client_sets(samples, labels, k=6):
    """{label: k nearest anchors} of every label a client declares; labels without samples get the set
    of all the client's sampled images together (as client_codes)."""
    if not samples:
        raise ValueError('image matching needs at least one image')
    S = anchor_matrix(next(iter(samples.values())).shape[-1])[1]
    sets = {x: anchor_set(v, k, S) for x, v in samples.items()}
    missing = [x for x in labels if x not in sets]
    if missing:
        fallback = anchor_set(np.concatenate(list(samples.values())), k, S)
        sets.update({x: fallback for x in missing})
    return {x: sets[x] for x in labels}
