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
