#!/usr/bin/env python3
"""Select coreset panels: representative Mini panels and discriminative Delta panels.

Because the benchmark macro-averages over languages with equal weight, selection
**decomposes per language** -- each language is an independent stratified-sampling
problem, and its per-language budget is a free choice. Every selector here therefore
works inside one language and returns (item indices, weights) whose weights sum to 1.

Selectors (all bake off against each other in validate.py):

  random_global     k items drawn uniformly from the whole set, ignoring language.
                    The naive baseline -- and it can miss small languages entirely.
  random_perlang    k_l items drawn uniformly inside each language.
  strat_meta        stratify on metadata only (length tercile x content flag, plus the
                    prompt-quality bin on the in-the-wild set), draw at random in-cell.
                    THIS IS THE SHIPPED SELECTOR for Mini panels.
  strat_quantile    equal-count strata along the single denoised difficulty axis, drawn
                    at random -- the textbook design for a 1-D auxiliary variable.
  median_quantile   the same strata, taking each stratum's median item (deterministic).
  strat_response    metadata cells subdivided by *denoised response* clusters; one item
                    drawn at random per cluster, weighted by cluster mass.
  medoid_response   the same clusters, but take the medoid instead of drawing.
  medoid_raw        medoids of clusters built on RAW (un-denoised) response vectors --
                    isolates how much the low-rank denoising is worth.
  highvar           top-k by between-system variance. Discriminative but badly biased;
                    kept to demonstrate that, not because it is a candidate.
  greedy            forward selection directly minimizing error on the training systems.
  delta             discrimination-SNR selection for Delta panels (see below).

Measured outcome: at the shipped sizes every language-stratified *unbiased* selector ties
within noise, because held-out-cell CV finds the item response space is effectively
rank 1 -- items differ by overall difficulty, not by which system they favour, so there is
little for response-space clustering to exploit. What does separate clearly: ignoring the
language axis costs ~2x, `highvar` is catastrophic for the level, and denoising is worth
~3x for the medoid variants. `strat_meta` ships because it is unbiased, ties the best at
k>=2000, and defines its strata from *published metadata only* -- so an external user can
audit the design without access to this repo's historical results.

On medoids vs random draws: a medoid is *not* its cluster's mean, so medoid selection is
biased by construction, while drawing at random within a stratum is design-unbiased. The
unbiased variants lead; the medoid ones are measured, not assumed away. A fixed seed
keeps a published panel deterministic either way.

Delta panels optimize something different -- the paired difference between two nearby
checkpoints, not the level -- so they select on discrimination signal-to-noise with
redundancy control, and must never be reported as an absolute score.

    python scripts/coreset/panel_select.py --set clean --kind mini --selector strat_response -k 2000
"""

import argparse
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Fingerprints, Matrix, make_panel, save_panel  # noqa: E402

MIN_PER_LANG = 12          # Mini: keep every language individually reportable
# Delta panels are read only in aggregate as a paired difference, never per language, so
# they do not need a reportable per-language sample -- and with 35 languages a floor of 12
# would force >=420 items and make a small panel impossible. Diversity of languages still
# matters for the difference, hence a floor above 1.
MIN_PER_LANG_DELTA = 3
KMEANS_SEED = 20260717


# ---------------------------------------------------------------------------
# budget allocation across languages
# ---------------------------------------------------------------------------
def allocate(M, k_total, langs, train_si, mode="neyman", min_per_lang=MIN_PER_LANG):
    """Split the item budget across languages.

    Neyman allocation puts items where the per-utterance spread is largest, which is what
    minimizes the variance of the macro-average. A floor keeps every language reportable
    on its own, and no language is asked for more items than it has.
    """
    sizes = {lg: M.cols(lg).size for lg in langs}
    if mode == "equal":
        raw = {lg: 1.0 for lg in langs}
    elif mode == "proportional":
        raw = {lg: float(sizes[lg]) for lg in langs}
    else:
        # Neyman needs the *within-system* item spread: what drives the sampling error of
        # one system's language mean. Pooling all systems into one std instead would fold
        # in BETWEEN-system spread, which no amount of sampling affects -- a language where
        # cosyvoice3 scores 100 and zonos2 scores 8 would look enormously variable and get
        # a large budget it cannot use. Averaging the per-system variances removes that.
        raw = {}
        for lg in langs:
            cols = M.cols(lg)
            vs = []
            for si in train_si:
                y = M.m["wer_clip"][si, cols]
                y = y[np.isfinite(y)]
                if y.size > 1:
                    vs.append(float(y.var()))
            raw[lg] = float(np.sqrt(np.mean(vs))) if vs else 1.0
    tot = sum(raw.values()) or 1.0

    alloc = {lg: min(sizes[lg], max(min_per_lang, int(round(k_total * raw[lg] / tot))))
             for lg in langs}
    # Largest-remainder correction so the realized total lands on k_total.
    for _ in range(10000):
        diff = k_total - sum(alloc.values())
        if diff == 0:
            break
        if diff > 0:
            room = [lg for lg in langs if alloc[lg] < sizes[lg]]
            if not room:
                break
            for lg in sorted(room, key=lambda x: -raw[x])[:diff]:
                alloc[lg] += 1
        else:
            room = [lg for lg in langs if alloc[lg] > min_per_lang]
            if not room:
                break
            for lg in sorted(room, key=lambda x: raw[x])[:(-diff)]:
                alloc[lg] -= 1
    return alloc


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------
MIN_PER_CELL = 3   # below this, stratification costs more than the auxiliary is worth


def meta_cells(M, cols, budget=None):
    """Hard coverage cells: length tercile x content flag (x prompt-quality bin on ITW).

    Expressed as strata rather than as embedding weights so the guarantee is auditable:
    every non-empty cell keeps representation proportional to its true mass, whatever the
    response space thinks. Cells are the level at which quotas are enforced.

    The cell grid is **coarsened to fit the budget**. Stratification only pays when each
    cell gets several items; forcing a floor of one item into each of 30 cells with a
    48-item budget spends the whole budget on the floor and leaves nothing for
    proportional allocation -- which measurably *hurt* accuracy at k=1000 on the
    in-the-wild set before this was added. Dimensions are dropped in increasing order of
    importance: prompt quality is the strongest predictor here so it goes last, and the
    length tercile collapses to a median split before disappearing.
    """
    ntok = M.meta["n_ref_tok"][cols]
    q3 = np.nanpercentile(ntok, [33.3, 66.7])
    length3 = np.digitize(ntok, q3)
    length2 = np.digitize(ntok, np.nanpercentile(ntok, [50.0]))
    content = ((M.meta["has_digit"][cols] > 0) | (M.meta["has_abbrev"][cols] > 0)).astype(int)

    if M.set == "itw":
        pm = M.meta["prompt_msr"][cols]
        # The 5 uniform 1.0-4.0 bins the set was stratified on (stage B); NaN -> its own bin.
        qual = np.where(np.isfinite(pm),
                        np.clip(((np.nan_to_num(pm, nan=1.0) - 1.0) / 0.6).astype(int), 0, 4),
                        5)
    else:
        qual = np.zeros(len(cols), dtype=int)
    zero = np.zeros(len(cols), dtype=int)

    # Finest grid first; fall back until the budget can afford MIN_PER_CELL per cell.
    grids = [
        length3 * 100 + content * 10 + qual,
        length2 * 100 + content * 10 + qual,
        length2 * 100 + qual,
        length2 * 100 + content * 10 + zero,
        length2 * 100 + zero,
        zero,
    ]
    if budget is None:
        return grids[0]
    for g in grids:
        if budget >= MIN_PER_CELL * len(np.unique(g)):
            return g
    return grids[-1]


def largest_remainder(sizes, k):
    """Apportion k across cells proportionally to size, floor of 1 while budget allows."""
    keys = list(sizes)
    tot = sum(sizes.values()) or 1
    exact = {c: k * sizes[c] / tot for c in keys}
    out = {c: min(sizes[c], max(1, int(exact[c]))) for c in keys} if k >= len(keys) \
        else {c: 0 for c in keys}
    if k < len(keys):
        # Not enough budget for every cell: give one item to the k biggest cells.
        for c in sorted(keys, key=lambda x: -sizes[x])[:k]:
            out[c] = 1
        return out
    for _ in range(10000):
        diff = k - sum(out.values())
        if diff == 0:
            break
        if diff > 0:
            room = [c for c in keys if out[c] < sizes[c]]
            if not room:
                break
            for c in sorted(room, key=lambda x: -(exact[x] - out[x]))[:diff]:
                out[c] += 1
        else:
            room = [c for c in keys if out[c] > 1]
            if not room:
                break
            for c in sorted(room, key=lambda x: (exact[x] - out[x]))[:(-diff)]:
                out[c] -= 1
    return out


def _cluster(X, n, seed):
    """k-means labels for n clusters (identity mapping when n >= len(X))."""
    if n >= len(X):
        return np.arange(len(X))
    km = KMeans(n_clusters=n, n_init=4, random_state=seed)
    return km.fit_predict(np.nan_to_num(X, nan=0.0))


# ---------------------------------------------------------------------------
# selectors -- each returns (local_indices_into_cols, weights summing to 1)
# ---------------------------------------------------------------------------
def _raw_response(M, cols, train_si):
    """Standardized raw response block [n_items, n_systems*n_metrics], NaN-filled."""
    blocks = []
    for metric in ("wer_clip", "spk", "msr"):
        use = [si for si in train_si if not (metric == "spk" and M.preset[si])]
        Y = M.m[metric][np.ix_(use, cols)].astype(np.float64)
        mu = np.nanmean(Y, axis=1, keepdims=True)
        sd = np.nanstd(Y, axis=1, keepdims=True)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        blocks.append(np.nan_to_num((Y - np.nan_to_num(mu)) / sd).T)
    return np.hstack(blocks)


def sel_random(M, F, cols, k, rng, train_si, **_):
    idx = rng.choice(len(cols), size=min(k, len(cols)), replace=False)
    return idx, np.full(len(idx), 1.0 / len(idx))


def sel_highvar(M, F, cols, k, rng, train_si, **_):
    X = _raw_response(M, cols, train_si)
    var = X.var(axis=1)
    idx = np.argsort(-var)[:min(k, len(cols))]
    return idx, np.full(len(idx), 1.0 / len(idx))


def _cell_plan(M, cols, k):
    """Cell assignment + how many items each cell gets, coarsened to fit the budget."""
    k = min(k, len(cols))
    cells = meta_cells(M, cols, budget=k)
    uniq = {c: int((cells == c).sum()) for c in np.unique(cells)}
    return cells, largest_remainder(uniq, k)


def sel_strat_meta(M, F, cols, k, rng, train_si, **_):
    cells, plan = _cell_plan(M, cols, k)
    N = len(cols)
    idx, w = [], []
    for c, n_c in plan.items():
        if n_c <= 0:
            continue
        pool = np.flatnonzero(cells == c)
        take = rng.choice(pool, size=min(n_c, pool.size), replace=False)
        idx.extend(take.tolist())
        w.extend([(pool.size / N) / len(take)] * len(take))
    return np.array(idx, dtype=int), np.array(w)


def _strat_response(M, F, cols, k, rng, train_si, pick, use_raw=False):
    """Metadata cells subdivided by response clusters; cluster-mass weights.

    Weighting each selected item by |cluster| / N_lang is what makes the panel estimate
    the language mean rather than the mean of the selected items.
    """
    cells, plan = _cell_plan(M, cols, k)
    X = _raw_response(M, cols, train_si) if use_raw else F.fp[cols]
    N = len(cols)
    idx, w = [], []
    for c, n_c in plan.items():
        if n_c <= 0:
            continue
        pool = np.flatnonzero(cells == c)
        labels = _cluster(X[pool], n_c, KMEANS_SEED)
        for lab in np.unique(labels):
            members = pool[labels == lab]
            if pick == "medoid":
                Z = np.nan_to_num(X[members], nan=0.0)
                j = members[np.argmin(((Z - Z.mean(axis=0)) ** 2).sum(axis=1))]
            else:
                j = members[rng.integers(0, members.size)]
            idx.append(int(j))
            w.append(members.size / N)
    return np.array(idx, dtype=int), np.array(w)


def _strat_quantile(M, F, cols, k, rng, train_si, pick):
    """Equal-count strata along the single denoised difficulty axis.

    Cross-validated rank selection (denoise.py) lands on rank 1 in nearly every language:
    the item response space is effectively **one-dimensional** -- items differ by overall
    difficulty, not by which system they favour. For a 1-D auxiliary variable, textbook
    stratified sampling beats clustering: cutting it into equal-count strata gives near
    uniform weights, so the auxiliary's variance reduction is not given back as
    weight-induced variance (which is exactly what cluster-mass weighting does when the
    clusters come out unequal).
    """
    cells, plan = _cell_plan(M, cols, k)
    score = np.nan_to_num(F.fp[cols][:, 0], nan=0.0)
    N = len(cols)
    idx, w = [], []
    for c, n_c in plan.items():
        if n_c <= 0:
            continue
        pool = np.flatnonzero(cells == c)
        order = pool[np.argsort(score[pool], kind="stable")]
        for part in np.array_split(order, min(n_c, order.size)):
            if part.size == 0:
                continue
            j = part[part.size // 2] if pick == "median" else part[rng.integers(0, part.size)]
            idx.append(int(j))
            w.append(part.size / N)
    return np.array(idx, dtype=int), np.array(w)


def sel_strat_quantile(M, F, cols, k, rng, train_si, **_):
    return _strat_quantile(M, F, cols, k, rng, train_si, pick="random")


def sel_median_quantile(M, F, cols, k, rng, train_si, **_):
    return _strat_quantile(M, F, cols, k, rng, train_si, pick="median")


def sel_strat_response(M, F, cols, k, rng, train_si, **_):
    return _strat_response(M, F, cols, k, rng, train_si, pick="random")


def sel_medoid_response(M, F, cols, k, rng, train_si, **_):
    return _strat_response(M, F, cols, k, rng, train_si, pick="medoid")


def sel_medoid_raw(M, F, cols, k, rng, train_si, **_):
    return _strat_response(M, F, cols, k, rng, train_si, pick="medoid", use_raw=True)


def sel_greedy(M, F, cols, k, rng, train_si, **_):
    """Forward selection minimizing training-system error directly.

    Optimizes the thing we actually care about rather than a geometric proxy, but it can
    only see the training systems -- validate.py holds families out to check that it did
    not just memorize them.
    """
    k = min(k, len(cols))
    Ys, targets = [], []
    for metric in ("wer_clip", "spk", "msr"):
        use = [si for si in train_si if not (metric == "spk" and M.preset[si])]
        Y = M.m[metric][np.ix_(use, cols)].astype(np.float64)
        scale = np.nanstd(Y) or 1.0
        Y = np.nan_to_num(Y, nan=float(np.nanmean(Y))) / scale
        Ys.append(Y)
        targets.append(Y.mean(axis=1))

    chosen = [int(rng.integers(0, len(cols)))]
    sums = [Y[:, chosen[0]].copy() for Y in Ys]
    for _ in range(k - 1):
        n = len(chosen) + 1
        best, best_j = np.inf, -1
        cost = np.zeros(len(cols))
        for Y, s, t in zip(Ys, sums, targets):
            cand = (s[:, None] + Y) / n           # [n_sys, n_items]
            cost += np.abs(cand - t[:, None]).mean(axis=0)
        cost[chosen] = np.inf
        best_j = int(np.argmin(cost))
        best = cost[best_j]
        if not np.isfinite(best):
            break
        chosen.append(best_j)
        sums = [s + Y[:, best_j] for Y, s in zip(Ys, sums)]
    idx = np.array(chosen, dtype=int)
    return idx, np.full(len(idx), 1.0 / len(idx))


def sel_delta(M, F, cols, k, rng, train_si, **_):
    """Delta panel: high discrimination SNR, de-duplicated in fingerprint space.

    Items where every system ties say nothing about a difference between two systems, and
    items whose apparent spread is noise only add variance to the paired estimate. Picks
    greedily by SNR while refusing near-duplicates of what is already in, so the panel
    covers distinct failure modes instead of one repeated.
    """
    k = min(k, len(cols))
    snr = F.snr[cols].copy()
    X = np.nan_to_num(F.fp[cols], nan=0.0)
    # Junk items -- where even ground truth fails ASR -- are dropped here (unlike the Mini
    # panels, which must keep them to stay unbiased for the full-set level).
    gt = M.m["wer_clip"][M.si("GT"), cols]
    snr[np.isfinite(gt) & (gt > 50)] = np.nan
    order = np.argsort(-np.nan_to_num(snr, nan=-np.inf))

    scale = np.linalg.norm(X, axis=1).mean() or 1.0
    chosen = []
    for j in order:
        if not np.isfinite(snr[j]):
            break
        if chosen:
            d = np.linalg.norm(X[chosen] - X[j], axis=1).min()
            if d < 0.25 * scale:
                continue
        chosen.append(int(j))
        if len(chosen) >= k:
            break
    if len(chosen) < k:  # redundancy filter too strict -- top up by SNR
        for j in order:
            if len(chosen) >= k:
                break
            if int(j) not in chosen and np.isfinite(snr[j]):
                chosen.append(int(j))
    idx = np.array(chosen, dtype=int)
    return idx, np.full(len(idx), 1.0 / max(len(idx), 1))


SELECTORS = {
    "random_perlang": sel_random,
    "strat_meta": sel_strat_meta,
    "strat_quantile": sel_strat_quantile,
    "median_quantile": sel_median_quantile,
    "strat_response": sel_strat_response,
    "medoid_response": sel_medoid_response,
    "medoid_raw": sel_medoid_raw,
    "highvar": sel_highvar,
    "greedy": sel_greedy,
    "delta": sel_delta,
}


def build_panel(M, F, selector, k, kind="mini", seed=0, train_si=None, langs=None,
                alloc_mode="neyman", min_per_lang=None):
    """Run a selector across all languages and assemble a panel record."""
    if min_per_lang is None:
        min_per_lang = MIN_PER_LANG_DELTA if kind == "delta" else MIN_PER_LANG
    langs = langs or M.langs
    train_si = train_si if train_si is not None else [i for i, r in enumerate(M.roles)
                                                      if r == "system"]
    rng = np.random.default_rng(seed)

    if selector == "random_global":
        # The one selector that does not respect the language axis -- sample globally and
        # let the chips fall, which is exactly the failure mode worth showing.
        all_cols = np.concatenate([M.cols(lg) for lg in langs])
        pick = rng.choice(all_cols.size, size=min(k, all_cols.size), replace=False)
        gidx = all_cols[pick]
        per_lang = {}
        for lg in langs:
            sel = gidx[M.lang[gidx] == lg]
            if sel.size:
                per_lang[lg] = {"idx": sel.tolist(),
                                "w": [1.0 / sel.size] * sel.size}
        return make_panel(M.set, kind, selector, k, per_lang,
                          {"seed": seed, "alloc": "global",
                           "langs_missing": [lg for lg in langs if lg not in per_lang]})

    fn = SELECTORS[selector]
    alloc = allocate(M, k, langs, train_si, mode=alloc_mode, min_per_lang=min_per_lang)
    per_lang = {}
    for lg in langs:
        cols = M.cols(lg)
        idx_local, w = fn(M, F, cols, alloc[lg], rng, train_si)
        if idx_local.size == 0:
            continue
        w = np.asarray(w, dtype=float)
        w = w / w.sum()                       # guard against rounding drift
        per_lang[lg] = {"idx": cols[idx_local].tolist(), "w": w.tolist()}
    return make_panel(M.set, kind, selector, k, per_lang,
                      {"seed": seed, "alloc": alloc_mode, "min_per_lang": min_per_lang})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["clean", "itw"])
    ap.add_argument("--selector", default="strat_response",
                    choices=sorted(SELECTORS) + ["random_global"])
    ap.add_argument("--kind", default="mini", choices=["mini", "delta"])
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alloc", default="neyman", choices=["neyman", "equal", "proportional"])
    args = ap.parse_args()

    M = Matrix(args.set)
    F = Fingerprints(args.set, matrix=M)  # fail loudly on a stale fp_<set>.npz
    panel = build_panel(M, F, args.selector, args.k, kind=args.kind, seed=args.seed,
                        alloc_mode=args.alloc)
    path = save_panel(panel)
    print(f"{path}\n  {panel['k_actual']} items across {len(panel['langs'])} languages")


if __name__ == "__main__":
    main()
