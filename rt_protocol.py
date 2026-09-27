"""
RT (relation-table) protocol, run ONCE before FL training.

psi_fl_relation_table.tex (labels -> rendered numbers: a1=1, a2=2, a3=3, a3b=4, a4=5, gid=6)

method="filter"  (default) Alg.2 fuzzy description PSI  ->  Alg.5 (\\label{alg:a4}) image-set
                 affinity filter. A pair survives only if BOTH signals agree.
method="affscan" Alg.3: affinity profile over public probes as the only signal.
method="attn"    Alg.4 (\\label{alg:a3b}) on DESCRIPTIONS: r_i(a)=enc_i(description of a) attended over the
                 PUBLIC TEXT anchor word list (TEXT_ANCHORS). Each client embeds the anchor words with ITS
                 OWN encoder (K_i = enc_i(anchors), clients[i]["anchor_vecs"]), so the only thing agreed in
                 advance is the anchor LIST: alpha is indexed by anchor, comparable across clients even when
                 their encoders / languages differ. Corrected log-attention coordinates
                 -> fuzzy PSI -> confident Mutual. No images.
method="attn_filter"  positive text ranks gate image PSI; normalized rank fusion requires
                 highest-rung support in at least one modality, then Mutual. This extends
                 the original cascade: attn_match="intersection" restores Alg.5 filtering.
Aggregator       alg:gid union-find (same-client constraint).

LabeledFuzzyPSI (psi="he"): CKKS / TenSEAL, semi-honest.
  receiver j : own keypair; publishes context (no secret key) + Enc(s_j), column-packed.
  sender   i : Enc(<s_i[a], s_j[b]>) homomorphically; per rung delta returns
               Enc((<.,.> - delta) * r), r ~ U(1,1000) fresh per slot, per rung, per pair.
               Slots outside the allowed mask (filter: classes not referenced in E') get r = 0.
  receiver j : decrypts, keeps only the sign -> bits -> rank R, sends R back to i.
  Leaks: R, |C_i|, and the masked magnitudes to j (quantified by the diagnostics: v_est_err).
psi="plain": same bits without crypto (debug / reference).

Rows of every signal matrix are ordered by masked id: row index == masked id.
"""
import zlib
from collections import Counter, defaultdict

import numpy as np

# public text anchor bank for method attn*: generic category words; deliberately NO number words / digits
TEXT_ANCHORS = (
    "animal mammal bird fish insect reptile amphibian pet dog cat horse cow sheep pig goat deer bear wolf fox "
    "lion tiger elephant monkey rabbit mouse rat squirrel whale dolphin shark eagle owl parrot duck chicken frog "
    "snake turtle lizard butterfly bee spider ant plant tree flower grass leaf fruit apple orange banana grape "
    "vegetable carrot potato mushroom seed forest mountain river lake ocean sea beach island desert sky cloud "
    "rain snow sun moon star fire water rock sand road bridge building house tower castle church city street "
    "vehicle car truck bus train airplane ship boat bicycle motorcycle tractor rocket tank wheel engine "
    "furniture chair table bed sofa lamp clock door window shelf tool hammer knife scissors key bottle cup "
    "plate bowl spoon phone computer keyboard television camera radio machine device clothing shirt dress "
    "trousers coat jacket shoe boot sandal sneaker bag hat glove sock uniform food bread cake pizza meat cheese "
    "person man woman child baby girl boy face hand eye head body sport ball game toy music instrument "
    "guitar piano drum book paper pen letter alphabet symbol character sign word text script handwriting "
    "mark stroke line curve circle loop oval ring hook cross dot square triangle angle corner edge shape "
    "vertical horizontal diagonal straight bent round open closed thin thick small large tall short wide "
    "narrow upper lower top bottom left right center pattern texture color black white red green blue yellow "
    "light dark bright metal wood glass plastic stone cloth leather paint drawing picture photo map logo "
    "vowel consonant capital lowercase uppercase quantity amount count measure size weight length speed "
    "time day night season winter summer spring autumn weather home school office shop market hospital "
    "farm garden park kitchen room wall floor roof field space planet earth energy power"
).split()

LADDER = np.round(np.arange(0.90, 0.40 - 1e-9, -0.05), 2)
# Precision preprocessing spends more PSI comparisons to resolve near-equal
# image prototypes. Values are public policy parameters, not fitted from gt.
VERIFY_LADDER = np.unique(np.round(np.r_[np.arange(.5, .90, .05), np.arange(.90, 1.0, .002)], 6))


def rank_bounds(R, ladder):
    """Conservative similarity bounds implied solely by strict PSI comparisons."""
    thresholds = np.sort(np.asarray(ladder, float))
    if not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError("a nonempty finite ladder is required")
    return np.r_[-1., thresholds][R], np.r_[thresholds, 1.][R]


def verified_candidates(text_rank, image_rank, verify_rank, text_ladder, image_ladder,
                        verify_ladder=VERIFY_LADDER, floor=.90, margin=.01):
    """Abstaining pair verification from quantized bounds, never raw cosines.

    A strong text or image signal nominates a pair. Shared-encoder full image
    features must also support it. Weak-text image rescues require a provable
    bidirectional image-feature margin over ALL rivals, including rejected
    candidates. Strong text can resolve image ambiguity but cannot override
    an image feature score below floor or a clearly stronger image rival.
    """
    tl, _ = rank_bounds(text_rank, text_ladder)
    il, _ = rank_bounds(image_rank, image_ladder)
    vl, vu = rank_bounds(verify_rank, verify_ladder)
    strong_text = tl >= max(text_ladder) - 1e-10
    strong_image = il >= max(image_ladder) - 1e-10
    eligible = ((text_rank > 0) & (image_rank > 0) & (strong_text | strong_image)
                & (vl >= floor - 1e-10))
    supported = np.zeros_like(eligible)
    for a, b in zip(*np.where(eligible)):
        rivals = np.r_[np.delete(vu[a], b), np.delete(vu[:, b], a)]
        competitor = float(rivals.max()) if len(rivals) else -1.
        if strong_text[a, b]:
            supported[a, b] = vl[a, b] >= competitor - margin
        else:
            supported[a, b] = vl[a, b] >= competitor + margin
    # Image verification receives equal weight to the two earlier modalities.
    score = (tl + il + vl) / 3
    return supported, np.where(supported, score, 0.)

HE_POLY = 8192
HE_COEFF = [60, 40, 40, 60]
HE_SCALE = 2 ** 40
MASK_RANGE = (1.0, 1000.0)
DIAG_MAX = 200_000                      # cap on stored diagnostic samples per array


# ================================================================ shared subroutines
def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def center_unit(S):
    """rowmean removal (drops 'similar to everything' bias) then Unit."""
    return unit(S - S.mean(axis=1, keepdims=True))


def one_sided(R):
    M = {}
    for a, row in enumerate(R):
        m = row.max()
        if m == 0 or (row == m).sum() > 1:
            continue
        M[a] = int(row.argmax())
    return M


def mutual(R):
    vi, vj = one_sided(R), one_sided(R.T)
    return [(a, b) for a, b in vi.items() if vj.get(b) == a]


def n_ties(R):
    """rows (both sides) dropped by OneSided because of a tied non-zero max."""
    c = 0
    for M in (R, R.T):
        for row in M:
            m = row.max()
            c += int(m > 0 and (row == m).sum() > 1)
    return c


def _head_split(X, n_heads):
    """Split [n, d] into [n, h, d/h] without changing the client-local space."""
    if X.ndim != 2:
        raise ValueError(f"attention input must be 2-D, got shape {X.shape}")
    n, d = X.shape
    if d % n_heads != 0:
        raise ValueError(f"attention dimension {d} must be divisible by n_heads={n_heads}")
    return X.reshape(n, n_heads, d // n_heads)


def attn_coord(q, K, V, tau, n_heads=1):
    """
    Multi-head cross-attention from descriptions to public text anchors.

    Q = description embeddings, K/V = public anchor embeddings.  Each head
    attends in a separate subspace, and the weighted anchor values are
    concatenated back into the original embedding dimension.

    Alpha uses shared anchor indices; the default PSI representation corrects
    its centered logarithm for correlations between public anchors.
    Pooled values remain in the client's encoder space. No learned
    projection matrices are introduced here because this protocol is currently
    numpy-only and has no training phase.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("attention temperature tau must be positive")
    if not isinstance(n_heads, (int, np.integer)) or n_heads < 1:
        raise ValueError("n_heads must be a positive integer")
    if any(x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all() for x in (q, K, V)):
        raise ValueError("Q, K, V must be nonempty finite 2-D matrices")
    if len(K) != len(V):
        raise ValueError("K and V must have the same number of anchors")
    if q.shape[1] != K.shape[1] or K.shape[1] != V.shape[1]:
        raise ValueError(
            "Q, K, and V must have the same embedding dimension in this "
            "anchor-space implementation"
        )

    Qh = _head_split(q, n_heads)
    Kh = _head_split(K, n_heads)
    Vh = _head_split(V, n_heads)

    # Per-head cosine cross-attention. Each head is normalized independently
    # so the temperature has the same interpretation as the old cosine-based
    # implementation, while different heads attend in different subspaces.
    Qh = Qh / np.maximum(np.linalg.norm(Qh, axis=2, keepdims=True), 1e-12)
    Kh = Kh / np.maximum(np.linalg.norm(Kh, axis=2, keepdims=True), 1e-12)
    scores = np.einsum("qhd,ahd->qha", Qh, Kh) / tau
    scores = scores - scores.max(axis=2, keepdims=True)

    alpha = np.exp(scores)
    alpha /= np.maximum(alpha.sum(axis=2, keepdims=True), 1e-12)

    # Weighted values: [queries, heads, head_dim] -> [queries, embedding_dim].
    head_out = np.einsum("qha,ahd->qhd", alpha, Vh)
    pooled = head_out.reshape(len(q), -1)

    return alpha, pooled


def keyword_attn(c, labels, K, tau, n_heads=1):
    """Encode keywords BEFORE calling this function; never join them into sentences.

    keyword_vecs[a]: [keywords, local embedding dimension]. Optional keywords[a]
    provides text for local IDF weights, log(classes/document frequency).
    Context shared by every class gets zero weight, without labels
    from other clients or ground truth. Missing text means uniform weights.
    Legacy desc_vecs[a] is treated as a single query. Head probabilities are
    averaged: arbitrary local subspace indices are not shared coordinates.
    """
    vectors = c.get("keyword_vecs")
    terms = c.get("keywords") if vectors is not None else None
    df = Counter(t for a in labels for t in set(terms[a])) if terms is not None else None
    alphas, outputs, weights = [], [], []
    for a in labels:
        q = np.asarray(vectors[a] if vectors is not None else c["desc_vecs"][a], float)
        if vectors is None:
            q = q[None, :]
        alpha, pooled = attn_coord(q, K, K, tau, n_heads)
        w = np.ones(len(q))
        if terms is not None:
            if len(terms[a]) != len(q):
                raise ValueError("keywords and keyword_vecs must have matching lengths")
            # Duplicate keywords do not receive extra total weight.
            counts = Counter(terms[a])
            w = np.array([np.log(len(labels) / df[t]) / counts[t]
                          for t in terms[a]])
            if not w.any():  # one class or identical metadata: no evidence to prefer a term
                w = np.array([1 / counts[t] for t in terms[a]], dtype=float)
        w /= w.sum()
        alphas.append(np.einsum("k,kha->a", w, alpha) / n_heads)
        outputs.append(w @ pooled)
        weights.extend(w)
    return np.stack(alphas), np.stack(outputs), np.asarray(weights)


def anchor_log_coordinates(alpha, K):
    """Centered log attention, corrected for redundant public anchors.

    W=(Kc Kc.T)^(-1/2) in shared anchor coordinates, using a truncated
    pseudoinverse. Unlike U coordinates, U diag(1/s) U.T has no arbitrary
    SVD signs/rotations across clients. Only public anchors determine W.
    This preserves differences suppressed by peaked softmax probabilities;
    it does not learn semantics absent from the encoder. With one keyword
    it is mathematically a projection of that keyword into the anchor span.
    """
    Kc = unit(np.asarray(K, float))
    Kc = Kc - Kc.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(Kc, full_matrices=False)
    keep = s > max(s[0] * 1e-6, 1e-12)
    W = (U[:, keep] / s[keep]) @ U[:, keep].T
    log_alpha = np.log(np.maximum(alpha, np.finfo(float).tiny))
    return unit((log_alpha - log_alpha.mean(axis=1, keepdims=True)) @ W)


def fused_rank(text_rank, image_rank, text_rungs, image_rungs, confidence="top"):
    """Equal-weight normalized rank fusion, requiring positive evidence in both.

    Keep integer arithmetic: this is proportional to mean normalized ranks,
    so unequal ladder lengths cannot silently change the modality weights.
    At least one modality must exceed its highest public ladder threshold
    (confidence=top); positive reproduces permissive fusion for ablations.
    Mutual matching happens AFTER fusion. Only released PSI ranks are used.
    """
    allowed = (text_rank > 0) & (image_rank > 0)
    if confidence == "top":
        allowed &= (text_rank == text_rungs) | (image_rank == image_rungs)
    return (text_rank.astype(np.int64) * image_rungs +
            image_rank.astype(np.int64) * text_rungs) * allowed


# ================================================================ Phase-1 signals (local)
def desc_embed(names, dim=128):
    """enc_i for descriptions: char 1-3-gram hashing (case-sensitive; no download).
    No padding spaces: padded grams gave every single-char name a shared baseline (cos exactly 0.5 =
    a ladder rung) -> 34% CKKS/plain disagreement on exact ties in the smoke run."""
    T = np.zeros((len(names), dim))
    for r, s in enumerate(names):
        s = str(s)
        for n in (1, 2, 3):
            for k in range(len(s) - n + 1):
                T[r, zlib.crc32(s[k:k + n].encode()) % dim] += 1.0
    return unit(T)


def aff_profile(R, P):
    """aff_i(a,p) = cos(prototype of class a, probe p) in the public encoder space. [n, |P|]"""
    return unit(R) @ unit(P).T


# ================================================================ LabeledFuzzyPSI
def fpsi_rank_plain(si, sj, ladder=LADDER, mask=None):
    R = sum((si @ sj.T > d).astype(np.int32) for d in ladder)   # '>' = HE sign test
    return R if mask is None else R * mask


class FPSIReceiver:
    """Party j. Holds the CKKS secret key; publishes context + Enc(s_j)."""

    def __init__(self, sj):
        import tenseal as ts
        self.ts = ts
        self.n, self.d = sj.shape
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, HE_POLY, coeff_mod_bit_sizes=HE_COEFF)
        self.ctx.global_scale = HE_SCALE
        self.reps = (HE_POLY // 2) // self.n
        if self.reps == 0:
            raise ValueError(f"{self.n} classes > {HE_POLY // 2} slots; raise HE_POLY")
        cts = [ts.ckks_vector(self.ctx, np.tile(sj[:, k], self.reps).tolist()) for k in range(self.d)]
        self.message = {"context": self.ctx.serialize(save_secret_key=False),
                        "columns": [c.serialize() for c in cts], "n": self.n, "reps": self.reps}

    def rank(self, reply, n_i, n_rungs):
        """-> (R [n_i, n_j], raw masked values U [n_rungs, n_i, n_j]) ; U only for diagnostics."""
        R = np.zeros((n_i, self.n), dtype=np.int32)
        U = np.zeros((n_rungs, n_i, self.n))
        it = iter(reply)
        for start in range(0, n_i, self.reps):
            m = min(self.reps, n_i - start)
            for k in range(n_rungs):
                v = np.asarray(self.ts.ckks_vector_from(self.ctx, next(it)).decrypt())[: m * self.n]
                v = v.reshape(m, self.n)
                U[k, start:start + m] = v
                R[start:start + m] += v > 0
        return R, U


class FPSISender:
    """Party i's view of one receiver: only the public message."""

    def __init__(self, message):
        import tenseal as ts
        self.ctx = ts.context_from(message["context"])
        assert not self.ctx.has_secret_key()
        self.cols = [ts.ckks_vector_from(self.ctx, b) for b in message["columns"]]
        self.nj, self.reps = message["n"], message["reps"]

    def reply(self, si, ladder, rng, mask=None):
        out, size = [], self.reps * self.nj
        for start in range(0, len(si), self.reps):
            rows = si[start:start + self.reps]
            m = len(rows)
            acc = None
            for k, col in enumerate(self.cols):
                p = np.zeros(size)
                p[: m * self.nj] = np.repeat(rows[:, k], self.nj)
                t = col * p.tolist()
                acc = t if acc is None else acc + t
            allow = np.zeros(size)
            allow[: m * self.nj] = 1.0 if mask is None else mask[start:start + m].reshape(-1)
            for delta in ladder:
                r = rng.uniform(*MASK_RANGE, size) * allow      # fresh; 0 outside the mask
                out.append(((acc - float(delta)) * r.tolist()).serialize())
        return out


def v_estimate(U, ladder, lo=MASK_RANGE[0], hi=MASK_RANGE[1]):
    """What receiver j can infer about v=<x,y> from u_k=(v-delta_k)*r_k: interval intersection midpoint."""
    L = np.full(U.shape[1:], -np.inf)
    H = np.full(U.shape[1:], np.inf)
    for k, d in enumerate(ladder):
        u = U[k]
        pos = u > 0
        L = np.maximum(L, np.where(pos, d + u / hi, d + u / lo))
        H = np.minimum(H, np.where(pos, d + u / lo, d + u / hi))
    L = np.clip(L, -1, 1)
    H = np.clip(H, -1, 1)
    return (L + H) / 2


class _Diag:
    """Server-side instrumentation (uses ground truth; never influences the protocol)."""

    def __init__(self):
        self.d = defaultdict(list)
        self.scalars = Counter()

    def add(self, key, arr):
        arr = np.asarray(arr, dtype=np.float32).ravel()
        cur = sum(len(a) for a in self.d[key])
        if cur < DIAG_MAX:
            self.d[key].append(arr[: DIAG_MAX - cur])

    def arrays(self):
        return {k: np.concatenate(v) for k, v in self.d.items() if v}


def fpsi_all_pairs(sig, pairs, psi, ladder, rng, masks=None, diag=None, gt=None, tag=""):
    """Run LabeledFuzzyPSI for every pair (i<j) on one signal. Returns {(i,j): R}."""
    out = {}
    by_j = defaultdict(list)
    for i, j in pairs:
        by_j[j].append(i)
    for j, senders in by_j.items():
        sj = sig[j]
        if psi == "he":
            recv = FPSIReceiver(sj)
            send = FPSISender(recv.message)
        for i in senders:
            si = sig[i]
            mask = None if masks is None else masks[(i, j)]
            V = si @ sj.T                                   # plaintext, diagnostics only
            if psi == "he":
                R, U = recv.rank(send.reply(si, ladder, rng, mask), len(si), len(ladder))
                if mask is not None:
                    R = R * mask
                if diag is not None:
                    Rp = fpsi_rank_plain(si, sj, ladder, mask)
                    sel = np.ones_like(V, bool) if mask is None else mask.astype(bool)
                    dist = np.min(np.abs(V[None] - ladder[:, None, None]), axis=0)
                    diag.add(f"{tag}he_dist", dist[sel])
                    diag.add(f"{tag}he_mismatch", (R != Rp)[sel])
                    diag.add(f"{tag}v_est_err", (v_estimate(U, ladder) - V)[sel])
            else:
                R = fpsi_rank_plain(si, sj, ladder, mask)
            out[(i, j)] = R
            if diag is not None:
                diag.scalars[f"{tag}ties"] += n_ties(R)
                if gt is not None:
                    G = gt(i, j)
                    diag.add(f"{tag}cos_same", V[G])
                    diag.add(f"{tag}cos_diff", V[~G])
    return out


# ================================================================ Aggregator (alg:gid)
def global_table(edges, masked_lists, support=None):
    parent = {(i, m): (i, m) for i, ms in masked_lists.items() for m in ms}
    members = {k: {k[0]} for k in parent}
    nodes = {k: {k} for k in parent}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # ponytail: strongest rank first, so a conflict drops the weaker edge (tex leaves order open)
    for i, ma, j, mb, _ in sorted(edges, key=lambda e: -e[4]):
        ra, rb = find((i, ma)), find((j, mb))
        if ra == rb or members[ra] & members[rb]:
            continue
        if support is not None and any(frozenset((a, b)) not in support
                                       for a in nodes[ra] for b in nodes[rb]):
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
        nodes[ra] |= nodes.pop(rb)

    gid = {r: g for g, r in enumerate(sorted({find(k) for k in parent}))}
    return {k: gid[find(k)] for k in parent}


# ================================================================ full protocol
def run_rt_protocol(clients, P, method="filter", psi="he", ladder=LADDER, tau=None,
                    coord="log_whiten", n_heads=1, min_samples=10, seed=0, gt=None, log=print, K_text=None, masked_out=None,
                    attn_match="precision", attn_confidence="top", verify_ladder=VERIFY_LADDER,
                    verify_floor=.90, verify_margin=.01):
    """clients {i: {"summ": {a: r_i(a)}, "names": {a: str}, "count": {a: n}, "desc_vecs": {a: vec}}};
    P = public IMAGE anchor feats (centered); K_text = public TEXT anchor embeddings (attn methods).
    attn*: keyword_vecs[a] = individually encoded keywords, optional keywords[a]
    for local IDF aggregation. Legacy desc_vecs[a] is a single-query fallback.
    Default log_whiten/one head compares corrected log attention in anchor space.
    pooled requires aligned encoder spaces; multiple heads are an ablation.
    log_whiten requires one head. attn_match=fusion combines text/image ranks
    before Mutual; intersection reproduces the old attn_filter cascade and
    does not apply attn_confidence.
    Fusion requests image ranks for all positive text pairs, exposing more
    image ranks than the original cascade. No raw similarity tie-break is used.
    attn_confidence=top requires exceeding the highest ladder threshold in at
    least one modality (text alone for attn); positive is the old permissive rule.
    desc_vecs is optional for filter (else char n-grams).

    gt(i, a, j, b) -> bool is used ONLY for diagnostics.
    Returns (table {(i, a): gid}, edges, diag dict).
    """
    rng = np.random.default_rng(seed)
    diag = _Diag()
    lad = lambda k: np.asarray(ladder[k] if isinstance(ladder, dict) else ladder, dtype=float)
    text_attn = method in ("attn", "attn_filter")
    if text_attn:
        if coord not in ("alpha", "pooled", "log_whiten"):
            raise ValueError("coord must be alpha, pooled, or log_whiten")
        if attn_match not in ("precision", "fusion", "intersection"):
            raise ValueError("attn_match must be precision, fusion or intersection")
        if not -1 <= verify_floor <= 1 or not 0 <= verify_margin <= 2:
            raise ValueError("invalid verification floor or margin")
        if attn_confidence not in ("top", "positive"):
            raise ValueError("attn_confidence must be top or positive")
        if coord == "log_whiten" and n_heads != 1:
            raise ValueError("log_whiten requires n_heads=1")
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if K_text is None and not all("anchor_vecs" in c for c in clients.values()):
            raise ValueError("method attn*: give every client anchor_vecs (own embedding of TEXT_ANCHORS) or pass K_text")
        K_text = None if K_text is None else unit(np.asarray(K_text, float))
        tau = 0.05 if tau is None else tau
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("attention temperature tau must be positive")
        anchor_counts = {len(c.get("anchor_vecs", K_text)) for c in clients.values()}
        if len(anchor_counts) > 1:
            raise ValueError("clients must use the same ordered public anchor list")

    # masked ids: eligible classes (>= min_samples) first, in random order; row index == masked id
    inv, sigs, n_elig = {}, defaultdict(dict), {}
    for i, c in clients.items():
        elig = [a for a in sorted(c["summ"]) if c["count"][a] >= min_samples]
        rest = [a for a in sorted(c["count"]) if a not in set(elig)]   # below min / not sampled: own gid
        elig = [elig[k] for k in rng.permutation(len(elig))]
        inv[i] = {m: a for m, a in enumerate(elig + rest)}
        n = n_elig[i] = len(elig)
        if n == 0:
            continue
        if method != "attn":
            source = c.get("precision_summ", c["summ"]) if method == "attn_filter" and attn_match == "precision" else c["summ"]
            Rq = np.stack([source[a] for a in elig])
            prof = aff_profile(Rq, P)
            if method == "attn_filter" and attn_match == "precision":
                sigs["verify"][i] = unit(Rq)
            for row in prof:
                diag.add("aff_row_var", [row.var()])
        if text_attn:                           # Alg.4: multi-head cross-attention over public text anchors
            if "keyword_vecs" not in c and "desc_vecs" not in c:
                raise ValueError(f"method {method}: client {i} needs keyword_vecs or desc_vecs")
            Ki = unit(np.asarray(c["anchor_vecs"], float)) if "anchor_vecs" in c else K_text
            alpha, pooled, weights = keyword_attn(c, elig, Ki, tau, n_heads)
            diag.add("keyword_weight", weights)
            diag.add("attn_entropy", [
                float(np.mean(-np.sum(alpha * np.log(np.maximum(alpha, 1e-12)), axis=1)))
            ])
            sigs["main"][i] = (anchor_log_coordinates(alpha, Ki) if coord == "log_whiten"
                               else center_unit(alpha if coord == "alpha" else pooled))
        if method != "attn":                    # image signal: identical for filter / affscan / attn_filter
            sigs["aff"][i] = center_unit(prof)
        if method == "filter":                  # enc_i for descriptions: own vectors if given, else char n-grams
            if "desc_vecs" in c:
                sigs["desc"][i] = unit(np.stack([np.asarray(c["desc_vecs"][a], float) for a in elig]))
            else:
                sigs["desc"][i] = desc_embed([c["names"][a] for a in elig])

    ids = sorted(i for i in clients if n_elig[i] > 0)
    pairs = [(i, j) for x, i in enumerate(ids) for j in ids[x + 1:]]

    def G(i, j):                                      # ground-truth matrix in masked-row order
        return np.array([[gt(i, inv[i][a], j, inv[j][b]) for b in range(n_elig[j])]
                         for a in range(n_elig[i])], dtype=bool)
    g = G if gt is not None else None

    edges, support, review = [], None, []
    if method in ("attn", "affscan"):
        key = "main" if method == "attn" else "aff"
        Rs = fpsi_all_pairs(sigs[key], pairs, psi, lad(key), rng, diag=diag, gt=g, tag=f"{key}_")
        for (i, j), R in Rs.items():
            if method == "attn" and attn_confidence == "top":
                diag.scalars["low_confidence_pairs"] += int(((R > 0) & (R < len(lad(key)))).sum())
                R = R * (R == len(lad(key)))
            edges += [(i, a, j, b, int(R[a, b])) for a, b in mutual(R)]
    elif method == "attn_filter" and attn_match in ("fusion", "precision"):
        Rd = fpsi_all_pairs(sigs["main"], pairs, psi, lad("main"), rng, diag=diag, gt=g, tag="main_")
        masks = {p: (R > 0).astype(np.int32) for p, R in Rd.items()}
        live = [p for p in pairs if masks[p].any()]
        Ra = fpsi_all_pairs(sigs["aff"], live, psi, lad("aff"), rng, masks=masks,
                            diag=diag, gt=g, tag="aff_")
        diag.scalars["candidates"] = sum(int(m.sum()) for m in masks.values())
        if attn_match == "precision":
            # Compare all image-feature rivals: masking before margin checks
            # would manufacture confidence by hiding a competing class.
            Rv = fpsi_all_pairs(sigs["verify"], pairs, psi, np.asarray(verify_ladder), rng,
                                diag=diag, gt=g, tag="verify_")
            support = set()
        for i, j in live:
            if attn_match == "precision":
                accepted, R = verified_candidates(Rd[i, j], Ra[i, j], Rv[i, j], lad("main"), lad("aff"),
                                                   verify_ladder, verify_floor, verify_margin)
                for a, b in zip(*np.where(accepted)):
                    support.add(frozenset(((i, int(a)), (j, int(b)))))
                proposed = fused_rank(Rd[i, j], Ra[i, j], len(lad("main")), len(lad("aff"))) > 0
                review.extend((i, inv[i][int(a)], j, inv[j][int(b)], 1)
                              for a, b in zip(*np.where(proposed & ~accepted)))
                diag.scalars["verification_rejected"] += int(((Rd[i, j] > 0) & ~accepted).sum())
            else:
                R = fused_rank(Rd[i, j], Ra[i, j], len(lad("main")), len(lad("aff")), attn_confidence)
            diag.scalars["fused_ties"] += n_ties(R)
            matched = mutual(R)
            diag.scalars["image_resolved_pairs"] += len(set(matched) - set(mutual(Rd[i, j])))
            edges += [(i, a, j, b, float(R[a, b])) for a, b in matched]
        diag.scalars["filtered_out"] = diag.scalars["candidates"] - len(edges)
    else:                         # filter or legacy attn_filter intersection
        ck = "desc" if method == "filter" else "main"
        Rd = fpsi_all_pairs(sigs[ck], pairs, psi, lad(ck), rng, diag=diag, gt=g, tag=f"{ck}_")
        cand = {p: mutual(R) for p, R in Rd.items()}
        diag.scalars["candidates"] = sum(len(v) for v in cand.values())
        log(f"[RT] description PSI ({ck}): {diag.scalars['candidates']} candidate pairs")
        masks = {}
        for (i, j), E in cand.items():
            M = np.zeros((len(sigs['aff'][i]), len(sigs['aff'][j])), np.int32)
            ra, rb = {a for a, _ in E}, {b for _, b in E}
            M[np.ix_(sorted(ra), sorted(rb))] = 1
            masks[(i, j)] = M
        live = [p for p in pairs if cand[p]]
        Ra = fpsi_all_pairs(sigs["aff"], live, psi, lad("aff"), rng, masks=masks, diag=diag, gt=g, tag="aff_")
        for p in live:
            i, j = p
            keep = set(mutual(Ra[p]))
            edges += [(i, a, j, b, int(Rd[p][a, b] + Ra[p][a, b])) for a, b in cand[p] if (a, b) in keep]
        diag.scalars["filtered_out"] = diag.scalars["candidates"] - len(edges)
    log(f"[RT] {method}: {len(edges)} edges after PSI")

    if masked_out is not None:                    # local tables: (client, local id) -> masked id
        masked_out.update({(i, a): m for i in inv for m, a in inv[i].items()})
    table = global_table(edges, {i: list(inv[i]) for i in clients}, support=support)
    if support is not None:
        rejected = [e for e in edges if table[e[0], e[1]] != table[e[2], e[3]]]
        review.extend((i, inv[i][a], j, inv[j][b], 2) for i, a, j, b, _ in rejected)
        edges = [e for e in edges if table[e[0], e[1]] == table[e[2], e[3]]]
        diag.scalars["consistency_rejected_edges"] = len(rejected)
    out = {(i, inv[i][m]): gv for (i, m), gv in table.items()}
    d = diag.arrays()
    d.update({k: np.array(v) for k, v in diag.scalars.items()})
    if support is not None:
        # These are uncertain candidates, not ground-truth mistakes. Singleton
        # classes may also be genuinely unique to a client.
        d["review_pairs"] = np.asarray(review, dtype=np.int64).reshape(-1, 5)
        sizes = Counter(out.values())
        d["unmatched_classes"] = np.asarray([k for k, g in out.items() if sizes[g] == 1],
                                             dtype=np.int64).reshape(-1, 2)
    return out, edges, d


def to_group_map(table, client_group):
    """{(client, local id): gid} -> {group: {local id: gid}}.

    Majority vote per (group, local id), then injective (C10): if two local ids of one group
    pick the same gid, the one with more votes keeps it, the other gets a fresh gid.
    Global ids re-indexed to 0..G-1.
    """
    votes = defaultdict(Counter)
    for (i, a), gv in table.items():
        votes[(client_group[i], a)][gv] += 1
    picked = {k: c.most_common(1)[0] for k, c in votes.items()}      # (gid, n)
    owner = {}
    fresh = max(table.values(), default=-1) + 1
    final = {}
    for (grp, a), (gv, n) in sorted(picked.items(), key=lambda t: (-t[1][1], t[0])):
        if (grp, gv) in owner:
            gv, fresh = fresh, fresh + 1
        owner[(grp, gv)] = a
        final[(grp, a)] = gv
    dense = {gv: k for k, gv in enumerate(sorted(set(final.values())))}
    out = defaultdict(dict)
    for (grp, a), gv in final.items():
        out[grp][a] = dense[gv]
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def pair_metrics(assign, same):
    """assign {key: gid}; same(k1, k2) -> bool ground truth. Pairwise confusion over keys from
    different owners (key[0]). Returns dict in the repo's CSV column names."""
    keys = sorted(assign)
    TP = FP = TN = FN = 0
    for x, k1 in enumerate(keys):
        for k2 in keys[x + 1:]:
            if k1[0] == k2[0]:
                continue
            t, p = same(k1, k2), assign[k1] == assign[k2]
            TP += t and p; FP += (not t) and p; TN += (not t) and (not p); FN += t and (not p)
    rec = TP / (TP + FN) if TP + FN else 0.0
    spec = TN / (TN + FP) if TN + FP else 0.0
    prec = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = (TP * TN - FP * FN) / den if den else 0.0
    return {"recall": rec, "specificity": spec, "precision": prec,
            "average_accuracy": (rec + spec) / 2, "f1_score": f1, "mcc": mcc,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN}


if __name__ == "__main__":
    # self-check: 4 clients / 3 label spaces; concepts shared across spaces; names drive Alg.2.
    rng = np.random.default_rng(0)
    d = 64
    proto = rng.normal(size=(8, d))
    P = rng.normal(size=(200, d)) * 3
    P = P - P.mean(0)
    concepts = {"A": [0, 1, 2, 3], "B": [2, 3, 4, 5], "C": [0, 5, 6, 7]}
    word = ["zero", "one", "two", "three", "four", "five", "six", "seven"]
    clients_ds = {0: "A", 1: "A", 2: "B", 3: "C"}
    clients = {c: {"summ": {l: proto[k] * 3 + rng.normal(size=d) * 0.3 for l, k in enumerate(concepts[g])},
                   "names": {l: word[k] for l, k in enumerate(concepts[g])},
                   "count": {l: 50 for l in range(4)}}
               for c, g in clients_ds.items()}
    # decoy: client 2 swaps names of concepts 4/5 -> desc says (2:'five') ~ (3:'five'), images disagree
    clients[2]["names"][2], clients[2]["names"][3] = "five", "four"
    concept = lambda i, a: concepts[clients_ds[i]][a]
    gt = lambda i, a, j, b: concept(i, a) == concept(j, b)

    # HE rank == plaintext rank, with and without mask
    s0 = center_unit(aff_profile(np.stack(list(clients[0]["summ"].values())), P))
    s2 = center_unit(aff_profile(np.stack(list(clients[2]["summ"].values())), P))
    recv = FPSIReceiver(s2)
    mask = np.array([[1, 0, 1, 1]] * 4)
    for mk in (None, mask):
        R, _ = recv.rank(FPSISender(recv.message).reply(s0, LADDER, rng, mk), 4, len(LADDER))
        R = R if mk is None else R * mk
        assert (R == fpsi_rank_plain(s0, s2, LADDER, mk)).all()

    # text side for attn*: concept embeddings; "description" of concept k = its vector + noise
    tvec = rng.normal(size=(8, 32))
    K_text = rng.normal(size=(150, 32))
    for c, g in clients_ds.items():
        clients[c]["desc_vecs"] = {l: tvec[k] + rng.normal(size=32) * 0.2 for l, k in enumerate(concepts[g])}
    # decoy for attn_filter: client 2's descriptions of concepts 4/5 swapped (images disagree)
    clients[2]["desc_vecs"][2], clients[2]["desc_vecs"][3] = clients[2]["desc_vecs"][3], clients[2]["desc_vecs"][2]

    # client 3 uses a DIFFERENT text encoder (random orthogonal rotation of the space): only the anchor
    # LIST is shared, each client embeds anchors + own descriptions with its own encoder
    Qrot, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    for c in clients:
        rot = Qrot if c == 3 else np.eye(32)
        clients[c]["anchor_vecs"] = K_text @ rot
        clients[c]["desc_vecs"] = {l: v @ rot for l, v in clients[c]["desc_vecs"].items()}
    K_text = None

    for method in ("filter", "affscan", "attn", "attn_filter"):
        for psi in ("plain", "he"):
            table, edges, dg = run_rt_protocol(clients, P, method=method, psi=psi, seed=1, gt=gt,
                                               log=lambda *_: None, K_text=K_text,
                                               ladder={"desc": LADDER, "aff": LADDER, "main": LADDER})
            m = pair_metrics(table, lambda k1, k2: gt(k1[0], k1[1], k2[0], k2[1]))
            if method == "filter":
                pass   # desc vectors of client 3 are rotated: direct comparison is meaningless (expected)
            elif method == "attn_filter":
                # decoy candidate must be killed by the image filter (FP=0); its recall loss is expected
                assert m["FP"] == 0 and dg["candidates"] > len(edges), (method, m, dg["candidates"], len(edges))
            elif method == "affscan":
                assert m["f1_score"] == 1.0, (method, m)
            else:   # attn uses descriptions only -> the decoy swap IS merged wrongly (that is what the filter fixes)
                assert m["FP"] > 0, (method, m)
            print(f"OK [{method:11s}|{psi:5s}] edges={len(edges):2d} F1={m['f1_score']:.3f} FP={m['FP']}")
