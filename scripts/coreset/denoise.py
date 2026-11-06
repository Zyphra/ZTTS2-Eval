#!/usr/bin/env python3
"""Turn raw item response vectors into denoised fingerprints for coreset selection.

An item's "behavioral fingerprint" is the column of the system x item matrix: which
systems did well or badly on it. Two items are redundant when they induce nearly the
same pattern across systems, even if their text looks nothing alike.

The catch is that with **one generation per (system, item) cell**, the item x system
interaction and the sampling noise are not separately identifiable. Clustering the raw
columns would therefore partly cluster on noise, and would select items whose apparent
discrimination is an artifact. This module handles that with low-rank shrinkage: fit a
rank-r model to the observed cells (r chosen by held-out-cell cross-validation) and use
the reconstruction, not the raw values, as the fingerprint. Everything the rank-r model
cannot explain is treated as noise.

Selection decomposes per language (the benchmark macro-averages over languages with
equal weight), so fingerprints only need to be comparable *within* a language. They are
computed and PCA-reduced per language.

Outputs ``assets/coreset/fp_<set>.npz``:

    fp        [N, fp_dim]  per-language PCA scores of the denoised response block
    snr       [N]          discrimination signal-to-noise, the Delta panel's selector
    sig_var   [N, n_metric] denoised between-system variance, per metric
    noise_var [N, n_metric] residual (unexplained) variance, per metric
    n_obs     [N]          how many system cells the item actually has

    python scripts/coreset/denoise.py --set clean --set itw
"""

import argparse
import os
import sys

import numpy as np

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

OUT_DIR = os.path.join(REPO, "assets", "coreset")

# Response metrics. spk is dropped for preset-voice engines, which do not clone the
# prompt speaker at all -- their spk cells carry no information about the item.
RESPONSE_METRICS = ("wer_clip", "spk", "msr")
MIN_OBS_SYSTEMS = 4      # below this an item's fingerprint is too sparse to trust
MAX_RANK = 6
CV_HOLDOUT = 0.10
CV_SEED = 20260717


def soft_impute_lowrank(Y, mask, rank, n_iter=30):
    """Rank-`rank` reconstruction of Y over observed cells, by iterative SVD imputation.

    Y is [M, N] with arbitrary values where ``mask`` is False. Missing cells start at the
    item mean over observed systems (0 after centering) and are refilled from the current
    reconstruction each pass, which converges to a rank-r fit of the observed entries.
    """
    X = np.where(mask, Y, 0.0)
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, s.size)
        Xr = (U[:, :r] * s[:r]) @ Vt[:r]
        new = np.where(mask, Y, Xr)
        if np.allclose(new, X, rtol=1e-5, atol=1e-6):
            X = new
            break
        X = new
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, s.size)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def choose_rank(Y, mask, rng, max_rank=MAX_RANK):
    """Pick the reconstruction rank by held-out-cell cross-validation.

    Hides a random CV_HOLDOUT share of observed cells, fits each candidate rank on the
    rest, and scores squared error on the hidden cells. The rank that generalizes best is
    the one whose structure is real rather than fitted noise.
    """
    obs = np.argwhere(mask)
    if len(obs) < 50:
        return 1, {}
    n_hold = max(10, int(len(obs) * CV_HOLDOUT))
    hold = obs[rng.choice(len(obs), size=n_hold, replace=False)]
    train_mask = mask.copy()
    train_mask[hold[:, 0], hold[:, 1]] = False
    if train_mask.sum() < 20:
        return 1, {}

    truth = Y[hold[:, 0], hold[:, 1]]
    errs = {}
    for r in range(1, min(max_rank, min(Y.shape) - 1) + 1):
        Xr = soft_impute_lowrank(Y, train_mask, r)
        pred = Xr[hold[:, 0], hold[:, 1]]
        errs[r] = float(np.mean((truth - pred) ** 2))
    best = min(errs, key=errs.get) if errs else 1
    return best, errs


def denoise_language(mats, sys_idx, preset, item_cols, rng, fp_dim):
    """Denoise one language's response block.

    Returns (fp [n_items, fp_dim], sig_var, noise_var, n_obs, ranks) where sig_var and
    noise_var are per (item, metric).
    """
    n_items = len(item_cols)
    n_met = len(RESPONSE_METRICS)
    sig = np.full((n_items, n_met), np.nan, dtype=np.float32)
    noise = np.full((n_items, n_met), np.nan, dtype=np.float32)
    blocks = []
    ranks = {}
    n_obs = np.zeros(n_items, dtype=np.int32)

    for mi, metric in enumerate(RESPONSE_METRICS):
        use = [si for si in sys_idx if not (metric == "spk" and preset[si])]
        Y = mats[metric][np.ix_(use, item_cols)].astype(np.float64)
        mask = ~np.isnan(Y)
        # Drop systems that scored almost nothing in this language -- an all-NaN row
        # contributes no structure and destabilizes the SVD.
        keep = mask.sum(axis=1) >= 0.5 * n_items
        if keep.sum() < MIN_OBS_SYSTEMS:
            continue
        Y, mask = Y[keep], mask[keep]

        # Standardize within (system, language): each system's row becomes zero-mean,
        # unit-variance, so a fingerprint compares systems on equal footing rather than
        # being dominated by whichever system has the widest raw spread.
        mu = np.array([r[m].mean() if m.any() else 0.0 for r, m in zip(Y, mask)])
        sd = np.array([r[m].std() if m.any() and r[m].std() > 0 else 1.0
                       for r, m in zip(Y, mask)])
        Z = (Y - mu[:, None]) / sd[:, None]
        Z = np.where(mask, Z, 0.0)

        rank, _errs = choose_rank(Z, mask, rng)
        ranks[metric] = rank
        Xr = soft_impute_lowrank(Z, mask, rank)

        resid = np.where(mask, Z - Xr, np.nan)
        sig[:, mi] = np.nanvar(Xr, axis=0)
        with np.errstate(invalid="ignore"):
            noise[:, mi] = np.nanvar(resid, axis=0)
        n_obs += mask.sum(axis=0).astype(np.int32)
        blocks.append(Xr)

    if not blocks:
        return (np.full((n_items, fp_dim), np.nan, dtype=np.float32),
                sig, noise, n_obs, ranks)

    # Item fingerprint = its column across every denoised metric block, PCA-reduced.
    F = np.vstack(blocks).T                       # [n_items, sum(M_metric)]
    F = F - F.mean(axis=0, keepdims=True)
    U, s, _Vt = np.linalg.svd(F, full_matrices=False)
    d = min(fp_dim, s.size)
    fp = np.full((n_items, fp_dim), np.nan, dtype=np.float32)
    fp[:, :d] = (U[:, :d] * s[:d]).astype(np.float32)
    return fp, sig, noise, n_obs, ranks


def run(set_name, fp_dim, verbose=True):
    d = np.load(os.path.join(OUT_DIR, f"matrix_{set_name}.npz"), allow_pickle=True)
    mats = {m: d[f"m_{m}"] for m in RESPONSE_METRICS}
    systems = d["systems"].tolist()
    roles = d["roles"].tolist()
    preset = d["preset"]
    lang = d["lang"]
    n_items = lang.size

    # Only real systems form the response space: GT/whisper are references, and the
    # partial-coverage rows are held back as novel-family validation subjects.
    sys_idx = [i for i, r in enumerate(roles) if r == "system"]
    if verbose:
        print(f"[{set_name}] response space: {len(sys_idx)} systems "
              f"({', '.join(systems[i] for i in sys_idx)})")

    fp = np.full((n_items, fp_dim), np.nan, dtype=np.float32)
    sig = np.full((n_items, len(RESPONSE_METRICS)), np.nan, dtype=np.float32)
    noise = np.full((n_items, len(RESPONSE_METRICS)), np.nan, dtype=np.float32)
    n_obs = np.zeros(n_items, dtype=np.int32)
    rng = np.random.default_rng(CV_SEED)

    for lg in sorted(set(lang.tolist())):
        cols = np.flatnonzero(lang == lg)
        f, s_, nz, nob, ranks = denoise_language(mats, sys_idx, preset, cols, rng, fp_dim)
        fp[cols], sig[cols], noise[cols], n_obs[cols] = f, s_, nz, nob
        if verbose:
            rr = " ".join(f"{k}=r{v}" for k, v in ranks.items())
            print(f"  {lg:4s} n={len(cols):5d}  {rr}")

    # Discrimination SNR: how much of the between-system spread on this item survives as
    # explainable structure, relative to what does not. This is the Delta panel's
    # selector -- items where every system ties carry no information about a difference,
    # and items whose spread is pure noise only add variance.
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nansum(sig, axis=1) / np.maximum(np.nansum(noise, axis=1), 1e-9)
    snr[n_obs < MIN_OBS_SYSTEMS] = np.nan

    out = os.path.join(OUT_DIR, f"fp_{set_name}.npz")
    np.savez_compressed(out, fp=fp, snr=snr.astype(np.float32), sig_var=sig,
                        noise_var=noise, n_obs=n_obs,
                        metrics=np.array(RESPONSE_METRICS), lang=lang, uid=d["uid"])
    if verbose:
        ok = np.isfinite(snr)
        print(f"[{set_name}] wrote {os.path.basename(out)}: {ok.sum()}/{n_items} items with "
              f"a usable fingerprint; median SNR {np.nanmedian(snr):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", choices=["clean", "itw"],
                    help="which set(s) (repeatable; default both)")
    ap.add_argument("--fp-dim", type=int, default=8, help="fingerprint dimensions (default 8)")
    args = ap.parse_args()
    for s in (args.sets or ["clean", "itw"]):
        run(s, args.fp_dim)


if __name__ == "__main__":
    main()
