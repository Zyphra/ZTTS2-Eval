"""Shared loading + estimation helpers for the coreset tooling.

The one thing every stage here must agree on is **the estimator**, so it lives in a
single place. ZTTS2-Eval's published score is a per-language mean over utterances
(`wers_clip` in utils/average_wer.py) followed by an *unweighted macro-average over languages*
(`macro` in scripts/results_doc/rd_common.py):

    S_m = (1/|L|) * sum_l  (1/N_l) * sum_{i in l} y_mi

so a panel estimates it with the matching weighted form, weights summing to 1 *within
each language*:

    S_hat_m = (1/|L|) * sum_l sum_{j in A_l} w_j * y_mj

Two consequences shape the whole design: selection decomposes per language (each is an
independent stratified-sampling problem), and per-language budget is a free choice
because language weights are equal regardless of N_l.
"""

import collections
import csv
import itertools
import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "assets", "coreset")
PANEL_DIR = os.path.join(OUT_DIR, "panels")

METRICS = ("wer", "wer_clip", "spk", "msr")
# Reported units, used for stating errors in the same numbers the RESULTS tables show.
METRIC_UNITS = {"wer": "pct", "wer_clip": "pct", "spk": "x100", "msr": "MOS"}
CI_BOOT = 10000
CI_SEED = 20260717


def weighted_boot(pieces, boot=CI_BOOT, seed=CI_SEED):
    """Draws of the weighted per-language bootstrap, the one resampling kernel every
    CI/SE here derives from.

    ``pieces`` is a list of ``(y, w)`` array pairs, one per language (``y`` may also be
    a paired difference). Each draw resamples items *within* each language -- the
    sampling unit -- then macro-averages the weighted means, matching the seeded
    bootstrap convention in utils/average_wer.py so panel CIs are comparable with the
    full-set ones. Callers take percentiles (CI) or the std (SE) of the returned draws.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(boot)
    for b in range(boot):
        acc = []
        for y, w in pieces:
            j = rng.integers(0, y.size, size=y.size)
            acc.append(np.sum(w[j] * y[j]) / np.sum(w[j]))
        draws[b] = np.mean(acc)
    return draws


class Matrix:
    """The system x item matrix plus the item metadata selection is constrained on."""

    def __init__(self, set_name):
        d = np.load(os.path.join(OUT_DIR, f"matrix_{set_name}.npz"), allow_pickle=True)
        self.set = set_name
        self.task = str(d["task"])
        self.systems = [str(x) for x in d["systems"]]
        self.families = [str(x) for x in d["families"]]
        self.roles = [str(x) for x in d["roles"]]
        self.preset = np.asarray(d["preset"], dtype=bool)
        self.lang = np.asarray([str(x) for x in d["lang"]])
        self.uid = np.asarray([str(x) for x in d["uid"]])
        self.m = {k: d[f"m_{k}"] for k in METRICS}
        self.meta = {f: d[f] for f in ("n_ref_tok", "n_chars", "has_digit", "has_abbrev",
                                       "is_question", "latin_frac", "prompt_dur_s",
                                       "prompt_msr")}
        self.langs = sorted(set(self.lang.tolist()))
        self._cols = {lg: np.flatnonzero(self.lang == lg) for lg in self.langs}

    def cols(self, lang):
        return self._cols[lang]

    def si(self, key):
        return self.systems.index(key)

    def subjects(self, include_partial=True):
        """Row indices of actual TTS systems (GT/whisper references excluded)."""
        ok = {"system", "partial"} if include_partial else {"system"}
        return [i for i, r in enumerate(self.roles) if r in ok]

    def covered_langs(self, si, metric="wer_clip", min_coverage=0.9):
        """Languages where this system scored at least `min_coverage` of the items.

        Partial leaves are kept in the matrix but must not be silently averaged into a
        macro-average as if complete -- a leaf that scored 55% of its language is a
        biased sample of it.
        """
        out = []
        for lg in self.langs:
            c = self.m[metric][si, self.cols(lg)]
            if np.isfinite(c).sum() >= min_coverage * c.size:
                out.append(lg)
        return out

    def true_score(self, si, metric, langs):
        """Full-benchmark macro-average for one system over the given languages."""
        vals = []
        for lg in langs:
            v = self.m[metric][si, self.cols(lg)]
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v.mean())
        return float(np.mean(vals)) if vals else float("nan")

    def panel_score(self, si, metric, panel):
        """Weighted panel estimate, using only the languages in the panel."""
        vals = []
        for lg, sel in panel["langs"].items():
            idx = np.asarray(sel["idx"], dtype=int)
            w = np.asarray(sel["w"], dtype=float)
            y = self.m[metric][si, idx]
            ok = np.isfinite(y)
            if not ok.any():
                continue
            # Renormalize over the items this system actually has, so a system with a
            # few failed generations is not silently scored as if they were zero.
            vals.append(float(np.sum(w[ok] * y[ok]) / np.sum(w[ok])))
        return float(np.mean(vals)) if vals else float("nan")

    def panel_score_ci(self, si, metric, panel, boot=CI_BOOT, seed=CI_SEED):
        """Weighted percentile-bootstrap CI95 of the panel estimate.

        Resamples items within each language (the sampling unit) via ``weighted_boot``.
        """
        per_lang = []
        for lg, sel in panel["langs"].items():
            idx = np.asarray(sel["idx"], dtype=int)
            w = np.asarray(sel["w"], dtype=float)
            y = self.m[metric][si, idx]
            ok = np.isfinite(y)
            if not ok.any():
                continue
            per_lang.append((y[ok], w[ok]))
        if not per_lang:
            return float("nan"), float("nan")
        draws = weighted_boot(per_lang, boot=boot, seed=seed)
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


class Fingerprints:
    def __init__(self, set_name, matrix=None):
        """Load fp_<set>.npz; pass the Matrix to assert the two files share an item axis.

        panel_select.py indexes ``fp`` with column indices from matrix_<set>.npz, so a
        stale fingerprint file silently misaligns every response-based selector. Callers
        that hold both should always pass ``matrix=``.
        """
        self.set = set_name
        d = np.load(os.path.join(OUT_DIR, f"fp_{set_name}.npz"), allow_pickle=True)
        self.fp = d["fp"]
        self.snr = d["snr"]
        self.sig_var = d["sig_var"]
        self.noise_var = d["noise_var"]
        self.n_obs = d["n_obs"]
        # Alignment keys (saved by denoise.py; absent from files written before then).
        self.lang = np.asarray([str(x) for x in d["lang"]]) if "lang" in d else None
        self.uid = np.asarray([str(x) for x in d["uid"]]) if "uid" in d else None
        if matrix is not None:
            self.check_against(matrix)

    def check_against(self, M):
        """Fail loudly when fp_<set>.npz does not describe the same items as the Matrix."""
        fix = f"rerun: python scripts/coreset/denoise.py --set {self.set}"
        if self.fp.shape[0] != len(M.uid):
            raise SystemExit(
                f"error: fp_{self.set}.npz has {self.fp.shape[0]} items but "
                f"matrix_{self.set}.npz has {len(M.uid)} — stale fingerprints; {fix}")
        if self.uid is not None and (
                not np.array_equal(self.uid, M.uid) or not np.array_equal(self.lang, M.lang)):
            raise SystemExit(
                f"error: fp_{self.set}.npz item axis (lang, uid) differs from "
                f"matrix_{self.set}.npz — stale fingerprints; {fix}")


def make_panel(set_name, kind, selector, k, per_lang, extra=None):
    """Assemble a panel record: per-language item indices and weights, plus provenance."""
    panel = {
        "set": set_name,
        "kind": kind,               # "mini" (estimates the level) | "delta" (differences only)
        "selector": selector,
        "k_requested": k,
        "k_actual": int(sum(len(v["idx"]) for v in per_lang.values())),
        "langs": per_lang,
    }
    if extra:
        panel.update(extra)
    return panel


def save_panel(panel, path=None):
    os.makedirs(PANEL_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(
            PANEL_DIR,
            f"{panel['set']}_{panel['kind']}_{panel['selector']}_k{panel['k_requested']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=1, sort_keys=True)
    return path


def load_panel(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# validation-CSV loading + aggregation, shared by validate.py (console) and
# gen_results_md.py (markdown) so the novel definition and gap edges live once
# ---------------------------------------------------------------------------
SIGN_GAP_EDGES = (0, 0.25, 0.5, 1, 2, 5, 10, np.inf)


def load_validation_rows(set_name):
    """Rows of ``validation_<set>.csv`` with fields typed and ``novel`` re-derived.

    The ``novel`` column is recomputed from (fold, family, role) rather than trusted: a
    system is novel when the selector never saw its family -- the held-out fold, plus
    every family that only ever appears as a partial-coverage row and is therefore never
    in any training pool. CSVs written before partial-only families were treated as
    novel have the stored flag wrong, and it is fully derivable from the file.

    Returns ``(rows, n_fixed)`` where n_fixed counts corrected flags. Raises
    FileNotFoundError when the CSV does not exist.
    """
    path = os.path.join(OUT_DIR, f"validation_{set_name}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for f in ("k", "seed", "novel", "k_actual", "n_langs"):
                r[f] = int(r[f])
            for f in ("est", "true", "abs_err", "worst_lang_err"):
                r[f] = float(r[f])
            rows.append(r)
    dense_fams = {r["family"] for r in rows if r["role"] == "system"}
    n_fixed = 0
    for r in rows:
        novel = int(r["family"] == r["fold"] or r["family"] not in dense_fams)
        n_fixed += int(novel != r["novel"])
        r["novel"] = novel
    return rows, n_fixed


def mae_cells(rows, metric, novel_only=True):
    """``{(selector, k): [abs_err, ...]}`` for one metric (novel-family rows only by
    default -- the held-out generalization readout)."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["metric"] == metric and (r["novel"] or not novel_only):
            by[(r["selector"], r["k"])].append(r["abs_err"])
    return by


def mean_se(vals):
    """(mean, standard error) of a list; SE is 0.0 for a single value."""
    se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), float(se)


def pairwise_sign_stats(rows, selector, k, metric="wer_clip", min_novel=1,
                        with_pairs=False):
    """Pooled pairwise sign accuracy of panel-estimated differences.

    Groups rows by (fold, seed) and takes all system pairs, keeping only pairs with at
    least ``min_novel`` genuinely held-out sides (default 1 -- a pair of two in-pool
    systems was fully visible to the selector, so it says nothing about held-out
    generalization; pass 0 for the pooled-everything view). Ties in the true score and
    self-pairs are skipped. Returns ``(gaps, correct)`` float arrays.

    The pooled pairs are NOT independent observations: the same two systems recur in
    every (fold, seed) group, so a gap bin holding 30 pooled pairs may contain only a
    handful of distinct comparisons re-drawn under different seeds. ``with_pairs=True``
    additionally returns an array of unordered system-pair labels ("a|b") aligned with
    ``gaps`` so callers can count the distinct pairs (clusters) behind each bin and flag
    bins too thin to read.
    """
    sel = [r for r in rows if r["selector"] == selector and r["k"] == k
           and r["metric"] == metric]
    groups = collections.defaultdict(list)
    for r in sel:
        groups[(r["fold"], r["seed"])].append(r)
    gaps, correct, pairs = [], [], []
    for g in groups.values():
        for a, b in itertools.combinations(g, 2):
            if a["system"] == b["system"] or a["novel"] + b["novel"] < min_novel:
                continue
            dt = a["true"] - b["true"]
            if dt == 0:
                continue
            gaps.append(abs(dt))
            correct.append(np.sign(dt) == np.sign(a["est"] - b["est"]))
            pairs.append("|".join(sorted((a["system"], b["system"]))))
    if with_pairs:
        return np.array(gaps), np.array(correct, dtype=float), np.array(pairs)
    return np.array(gaps), np.array(correct, dtype=float)


def check_weights(panel, tol=1e-6):
    """Every language's weights must sum to 1 -- otherwise the estimator is not the
    macro-average it claims to be. Returns the list of offending languages."""
    bad = []
    for lg, sel in panel["langs"].items():
        s = float(np.sum(sel["w"]))
        if abs(s - 1.0) > tol:
            bad.append((lg, s))
    return bad
