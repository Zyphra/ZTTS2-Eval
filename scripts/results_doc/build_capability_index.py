#!/usr/bin/env python3
"""Build the ZTTS2 capability and prosody indices: one score per system, Epoch-Capabilities-Index style.

Method (ECI, https://epoch.ai/data/eci-documentation/methodology): item response
theory with a logistic link,

    performance(s, i) = sigmoid(alpha_i * (C_s - D_i)),

one capability C_s per system and one (difficulty D_i, slope alpha_i) pair per item,
fit by ridge-penalised non-linear least squares over every observed (system, item)
score. One slope is pinned to fix the scale; the ridge term fixes the translation.

An item is one metric on one test set in one language (`wer|in-the-wild|de`).
A system scored on only some languages is still placed on the common scale, because
the items it does have carry their own fitted difficulty — the comparison a
macro-average over different language sets cannot make.

Every metric is mapped to a performance in [0, 1], higher = better (METRICS below).
Residuals are weighted so each quality dimension (intelligibility, voice, quality,
prosody) carries equal total weight, whatever its item count.

The prosody index is the same fit over prosody items only (PROSODY_METRICS) — ECI's
domain-specific index — with more of them than the capability index's prosody
dimension: the char/s pace factor (speaking_rate.json) and the paired In-The-Wild
duration / pause / pitch-range distances join MPM, rhythm and pitch range. Weights make
each prosody aspect (pace, pauses, pitch, contour) count equally, so a quantity
measured two ways is not counted twice. Same anchors, so both indices share one scale.

Scale: capabilities are mapped affinely so two anchor systems sit at fixed values
(ANCHORS). Confidence intervals: ECI's bootstrap — the system set is held fixed, each
system's observed items are resampled with replacement (keeping its item count), the
whole model is refit and re-anchored; 90% percentile interval.

Breakdown (breakdown()): per system, each dimension, metric and test set gets the level
its scores alone would place the system at and its share of the information the fit has
about it; weight × level summed over the dimensions (or over the sets) gives the index.

Reads the results snapshot (build_results.py) and the speaking-rate payload
(build_speaking_rate.py), and writes

    _cache/capability_index.json  ->  published to explorer/snapshot/ (publish_snapshot.py)

    python scripts/results_doc/build_capability_index.py [--bootstrap N]
"""
import argparse
import json
import os
import sys
import time

# The fit is many tiny vector ops: a BLAS thread pool in each bootstrap worker only
# oversubscribes the machine (minutes instead of seconds on a many-core node).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "_cache", "capability_index.json")
SPEAKING_RATE = os.path.join(SCRIPT_DIR, "_cache", "speaking_rate.json")

sys.path.insert(0, SCRIPT_DIR)
from rd_common import (  # noqa: E402
    PARAMS, RELEASES, SNAPSHOT_PATH, load_snapshot, rec_value, write_json_atomic)

TASKS = ("clean", "in-the-wild")
REAL = "real speech"   # the gt_<task> baselines, merged into one system

# metric -> (dimension, performance in [0, 1] (higher = better), what that is, task
# restriction or None). Speaker similarity on In-The-Wild is the held-out score (against
# a different real recording of the speaker), which copying the prompt's channel cannot
# inflate. The ×ceil prosody factors are 1 at the real-speech split-half ceiling.
METRICS = {
    "wer":           ("intelligibility", lambda v: 1 - min(v, 100) / 100, "1 − WER_CLIP/100", None),
    "tail50":        ("intelligibility", lambda v: 1 - v / 100, "1 − Tail>50%/100", None),
    "spk":           ("voice", lambda v: v / 100, "SPK prompt / 100", "clean"),
    "spk_xutt":      ("voice", lambda v: v / 100, "SPK held-out / 100", "in-the-wild"),
    "msr_utmos":     ("quality", lambda v: (v - 1) / 4, "(MSR-UTMOS − 1) / 4", None),
    "mpm_span":      ("prosody", lambda v: 1 - min(max(v, 0), 1), "1 − MPM span, clipped to [0, 1]", None),
    "rhythm_rate_x": ("prosody", lambda v: min(1, 1 / v), "min(1, 1 / rhythm pace ×ceil)", None),
    "rhythm_x":      ("prosody", lambda v: min(1, 1 / v), "min(1, 1 / rhythm pause ×ceil)", None),
    "pitch_range_x": ("prosody", lambda v: min(1, 1 / v), "min(1, 1 / pitch range ×ceil)", None),
}
DIMENSIONS = ("intelligibility", "voice", "quality", "prosody")

# The prosody index's items, by aspect. `charsec_x` comes from speaking_rate.json (d_ref
# over the split-half ceiling); the paired In-The-Wild distances are 0 when the
# generation matches the real recording, so they enter as 1 / (1 + distance), which the
# logistic fit reads as -log(distance), free of the metric's unit.
_X = lambda v: min(1, 1 / v)
PROSODY_METRICS = {
    "rhythm_rate_x": ("pace", _X, "min(1, 1 / rhythm pace ×ceil)", None),
    "charsec_x":     ("pace", _X, "min(1, 1 / char/s ×ceil)", None),
    "paired_dur":    ("pace", lambda v: 1 / (1 + v), "1 / (1 + paired Dur |log2|)", "in-the-wild"),
    "rhythm_x":      ("pauses", _X, "min(1, 1 / rhythm pause ×ceil)", None),
    "paired_pause":  ("pauses", lambda v: 1 / (1 + v), "1 / (1 + paired Pause Δ)", "in-the-wild"),
    "pitch_range_x": ("pitch", _X, "min(1, 1 / pitch range ×ceil)", None),
    "paired_range":  ("pitch", lambda v: 1 / (1 + v), "1 / (1 + paired Range |log2|)", "in-the-wild"),
    "mpm_span":      ("contour", lambda v: 1 - min(max(v, 0), 1), "1 − MPM span, clipped to [0, 1]", None),
}
ASPECTS = ("pace", "pauses", "pitch", "contour")

ANCHORS = ((REAL, 100.0), ("zonos2_accurate", 50.0))
RIDGE = 0.1
CLIP_EPS = 1e-3
CI_LEVEL = 0.90


def observations(rows, metrics=METRICS):
    """(system, item, dimension, performance) for every scored full-set leaf; `metrics`
    keys that the snapshot does not carry (charsec_x) are skipped here."""
    obs = []
    for r in rows:
        model, task = r["model"], r["task"]
        if model.startswith("gt_"):                 # baselines carry task "" in the snapshot
            task, model = model[3:], REAL
        if task not in TASKS or r["set"] == "ALL":
            continue
        for key, (dim, tf, _, only) in metrics.items():
            v = rec_value(r, key)
            if v is None or (only and only != task) or (key.endswith("_x") and v <= 0):
                continue
            obs.append((model, f"{key}|{task}|{r['set']}", dim, float(tf(v))))
    return obs


def scored_leaves(rows):
    """(system, set, language) of every full-set system leaf in the snapshot, i.e. the
    leaves that passed build_results' generation-coverage gate."""
    return {(r["model"], r["task"], r["set"]) for r in rows
            if r["task"] in TASKS and r["set"] != "ALL" and not r["model"].startswith("gt_")}


def charsec_observations(sr, leaves):
    """char/s pace items from the speaking-rate payload: one per (set, language, system).
    A system's item counts only where `leaves` (scored_leaves) has it: the speaking-rate
    payload takes any leaf with enough utterances, so it also carries partial generations
    that build_results drops for coverage, which would credit a system with languages
    it was never fully scored on. Real speech is the reference voice's split-half twin
    (`gt_*` with a split), the ceiling itself; a `gt_*` series without a split would be
    the reference scored against itself, so it is skipped."""
    dim, tf, _, _ = PROSODY_METRICS["charsec_x"]
    obs = []
    for p in sr["panels"]:
        if p["set"] not in TASKS:
            continue
        for s in p["series"]:
            v = s.get("x_ceiling")
            if v is None or v <= 0:
                continue
            if s["name"].startswith("gt_"):
                if not s.get("split"):
                    continue
                name = REAL
            elif (s["name"], p["set"], p["lang"]) in leaves:
                name = s["name"]
            else:
                continue
            obs.append((name, f"charsec_x|{p['set']}|{p['lang']}", dim, float(tf(v))))
    return obs


class Problem:
    """The design of one fit: index arrays + per-observation weights. Weights make each
    dimension's observations sum to the same total, whatever its item count."""

    def __init__(self, obs):
        self.systems = sorted({o[0] for o in obs})
        self.items = sorted({o[1] for o in obs})
        s_ix = {s: i for i, s in enumerate(self.systems)}
        i_ix = {it: i for i, it in enumerate(self.items)}
        self.si = np.array([s_ix[o[0]] for o in obs])
        self.ii = np.array([i_ix[o[1]] for o in obs])
        self.y = np.clip(np.array([o[3] for o in obs]), CLIP_EPS, 1 - CLIP_EPS)
        dims = np.array([o[2] for o in obs])
        present = sorted(set(dims))
        count = {d: int((dims == d).sum()) for d in present}
        self.w = np.array([len(obs) / (len(present) * count[d]) for d in dims])
        self.nS, self.nI = len(self.systems), len(self.items)
        self.anchor_item = 0                      # its slope is pinned to 1
        self.nP = self.nS + self.nI + (self.nI - 1)

    def unpack(self, p):
        C = p[:self.nS]
        D = p[self.nS:self.nS + self.nI]
        A = np.insert(p[self.nS + self.nI:], self.anchor_item, 1.0)
        return C, D, A

    def fit(self, si, ii, y, w, x0):
        """Minimise ECI's objective, ½·Σ w·(σ(α(C−D)) − y)² + ½·ridge·‖p‖²/n_params,
        with its analytic gradient (bounds: |C|, |D| ≤ 10, 0.1 ≤ α ≤ 10)."""
        nS, nI, nP, ai = self.nS, self.nI, self.nP, self.anchor_item

        def f(p):
            C, D, A = self.unpack(p)
            gap = C[si] - D[ii]
            s = 1 / (1 + np.exp(-np.clip(A[ii] * gap, -500, 500)))
            r = s - y
            e = w * r * s * (1 - s)              # d(½w·r²)/d(logit)
            gA = np.bincount(ii, e * gap, nI)
            g = np.concatenate([np.bincount(si, e * A[ii], nS),
                                -np.bincount(ii, e * A[ii], nI),
                                np.delete(gA, ai)])
            return (0.5 * np.sum(w * r * r) + 0.5 * RIDGE * np.sum(p * p) / nP,
                    g + RIDGE * p / nP)

        bounds = [(-10.0, 10.0)] * (nS + nI) + [(0.1, 10.0)] * (nI - 1)
        sol = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 50000, "maxfun": 100000, "ftol": 1e-13, "gtol": 1e-9})
        if not sol.success:
            raise RuntimeError(f"index fit did not converge: {sol.message}")
        return sol.x

    def x0(self):
        return np.concatenate([np.zeros(self.nS + self.nI), np.ones(self.nI - 1)])


def anchor_map(systems, C, anchors=ANCHORS):
    """(a, b) of index = a + b * C sending the two anchor systems to their values."""
    (hi_s, hi_v), (lo_s, lo_v) = anchors
    try:
        c_hi, c_lo = C[systems.index(hi_s)], C[systems.index(lo_s)]
    except ValueError as e:
        raise SystemExit(f"error: anchor system missing from the results: {e}")
    if not c_hi - c_lo > 1e-9:
        raise SystemExit(f"error: anchors do not define a scale ({hi_s} fitted at or below {lo_s})")
    b = (hi_v - lo_v) / (c_hi - c_lo)
    return lo_v - b * c_lo, b


def breakdown(P, x_hat, a, b, obs, metrics, dims):
    """Per system, how much each dimension, each of its metrics and each test set
    decides its index, and where each would put the system on its own.

    The fit places C_s where the gradient of the system's own weighted residuals
    vanishes (up to the tiny ridge term). Linearise each group g of those
    observations at C_s, Gauss-Newton style: H_g = Σ w·(σ'α)² is the information the
    group carries about C_s, g_g = Σ w·(σ − y)·σ'α its pull. Alone, the group would
    move C_s to C_s − g_g / H_g (its `level`), and since Σ_g g_g ≈ 0,

        C_s ≈ Σ_g (H_g / Σ H) · (C_s − g_g / H_g).

    Mapped to the index scale, `weight` = H_g / Σ H times `level` gives the group's
    `points`, which sum to the index. Items near the system's level and with steep
    slopes carry the most information, so equal residual weights per dimension do not
    make equal weights here. A group whose items are all saturated carries no
    information: weight 0, level None.
    """
    C, D, A = P.unpack(x_hat)
    dim = np.array([o[2] for o in obs])
    metric = np.array([o[1].split("|")[0] for o in obs])
    task = np.array([o[1].split("|")[1] for o in obs])
    z = A[P.ii] * (C[P.si] - D[P.ii])
    s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    J = s * (1 - s) * A[P.ii]
    G, H = P.w * (s - P.y) * J, P.w * J * J

    def part(rows, key):
        h, g = float(H[rows].sum()), float(G[rows].sum())
        level = a + b * (C[P.si[rows][0]] - g / h) if h > 0 else None
        return {"key": key, "h": h, "level": level, "n": int(rows.size)}

    def finish(p, total):
        w = p.pop("h") / total
        p["points"] = round(w * p["level"], 1) if p["level"] is not None else 0.0
        p["weight"] = round(w, 3)
        p["level"] = None if p["level"] is None else round(p["level"], 1)
        return p

    out = []
    for i in range(P.nS):
        own = P.si == i
        total = float(H[own].sum())
        groups = []
        for d in dims:
            rows = np.flatnonzero(own & (dim == d))
            if not rows.size:
                continue
            g = finish(part(rows, d), total)
            g["metrics"] = [finish(part(np.flatnonzero(own & (metric == m)), m), total)
                            for m in metrics if (own & (metric == m)).any() and metrics[m][0] == d]
            groups.append(g)
        sets = [finish(part(np.flatnonzero(own & (task == t)), t), total)
                for t in TASKS if (own & (task == t)).any()]
        out.append({"dimensions": groups, "sets": sets})
    return out


# --- bootstrap (worker processes) -------------------------------------------
_P = None


def _init_worker(problem):
    global _P
    _P = problem


def _draw(args):
    """One ECI bootstrap refit: hold the systems fixed, resample each system's own
    observations with replacement, refit, re-anchor. Returns that draw's index values."""
    seed, k, x_hat = args
    P = _P
    rng = np.random.default_rng([seed, k])
    idx = np.concatenate([rng.choice(rows, rows.size) for rows in
                          (np.flatnonzero(P.si == s) for s in range(P.nS))])
    C = P.unpack(P.fit(P.si[idx], P.ii[idx], P.y[idx], P.w[idx], x_hat))[0]
    a, b = anchor_map(P.systems, C)
    return a + b * C


def build_index(obs, metrics, dims, args):
    """Fit, anchor and bootstrap one index; its payload (method + ranked systems)."""
    P = Problem(obs)
    x_hat = P.fit(P.si, P.ii, P.y, P.w, P.x0())
    C = P.unpack(x_hat)[0]
    a, b = anchor_map(P.systems, C)
    index = a + b * C

    ci = [None] * P.nS
    if args.bootstrap > 0:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(args.seed, k, x_hat) for k in range(args.bootstrap)]
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(P,)) as ex:
            draws = np.array(list(ex.map(_draw, jobs, chunksize=4)))
        q = (1 - CI_LEVEL) / 2
        lo, hi = np.quantile(draws, [q, 1 - q], axis=0)
        anchors = {name for name, _ in ANCHORS}     # fixed by definition, not estimated
        ci = [None if s in anchors else [round(float(lo[i]), 1), round(float(hi[i]), 1)]
              for i, s in enumerate(P.systems)]

    langs = {}
    for s, item, dim, _ in obs:
        _, task, lang = item.split("|")
        langs.setdefault(s, {t: set() for t in TASKS})[task].add(lang)
    parts = breakdown(P, x_hat, a, b, obs, metrics, dims)
    systems = []
    for i in np.argsort(-index):
        s = P.systems[i]
        n = P.si == i
        systems.append({
            "name": s,
            "index": round(float(index[i]), 1),
            "ci": ci[i],
            "anchor": s in dict(ANCHORS),
            "n_items": int(n.sum()),
            "langs": {t: len(langs[s][t]) for t in TASKS},
            **({"release": RELEASES[s][0], "release_source": RELEASES[s][1]}
               if s in RELEASES else {}),
            **({"params": {"total": PARAMS[s][0], "active": PARAMS[s][1],
                           "source": PARAMS[s][2]}} if s in PARAMS else {}),
            "breakdown": parts[i],
        })
    n_langs = {t: len({it.split("|")[2] for it in P.items if it.split("|")[1] == t}) for t in TASKS}
    return {
        "method": {
            "model": "performance = sigmoid(slope_item * (capability_system - difficulty_item))",
            "reference": "Epoch Capabilities Index, https://epoch.ai/data/eci-documentation/methodology",
            "anchors": dict(ANCHORS),
            "dimensions": list(dims),
            "metrics": {k: {"dimension": d, "performance": desc, **({"set": only} if only else {})}
                        for k, (d, _, desc, only) in metrics.items()},
            "ridge": RIDGE,
            "ci_level": CI_LEVEL,
            "bootstrap": args.bootstrap,
            "n_obs": len(obs),
            "n_items": P.nI,
            "n_langs": n_langs,
        },
        "systems": systems,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=SNAPSHOT_PATH, help="results snapshot (build_results.py)")
    ap.add_argument("--speaking-rate", default=SPEAKING_RATE,
                    help="speaking-rate payload (build_speaking_rate.py), for the prosody index")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap refits for the CIs (0 = none)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rows = load_snapshot(args.snapshot)
    obs = observations(rows)
    if not obs:
        raise SystemExit(f"error: no scored full-set leaves in {args.snapshot}")
    if not os.path.isfile(args.speaking_rate):
        raise SystemExit(f"error: {args.speaking_rate} missing -- run build_speaking_rate.py first "
                         "(the prosody index needs its char/s pace items)")
    with open(args.speaking_rate, encoding="utf-8") as f:
        pros_obs = (observations(rows, PROSODY_METRICS)
                    + charsec_observations(json.load(f), scored_leaves(rows)))

    t0 = time.time()
    payload = build_index(obs, METRICS, DIMENSIONS, args)
    payload["prosody"] = build_index(pros_obs, PROSODY_METRICS, ASPECTS, args)
    write_json_atomic(args.out, payload, indent=1)
    for name, p in (("capability", payload), ("prosody", payload["prosody"])):
        m = p["method"]
        print(f"[capability_index] {name}: {len(p['systems'])} systems · {m['n_items']} items · "
              f"{m['n_obs']} scores")
    print(f"[capability_index] {args.bootstrap} bootstrap refits each · {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
