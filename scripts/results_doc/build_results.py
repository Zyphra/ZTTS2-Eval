#!/usr/bin/env python3
"""Build the explorer snapshot from core scores, MPM span, rhythm, pitch range and
paired ITW prosody; refuse inputs that are not ZTTS2-Eval <BENCHMARK_VERSION>."""
import collections, csv, os, re, json, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CACHE = os.path.join(SCRIPT_DIR, "_cache")
os.makedirs(CACHE, exist_ok=True)

ROOT = os.path.join(REPO_ROOT, "results")
DATA = os.path.join(REPO_ROOT, "data")
OUT_JSON = os.path.join(CACHE, "results_snapshot.json")

sys.path.insert(0, REPO_ROOT)
from utils.per_uid import CI_BOOT, CI_SEED, L50_THRESHOLD, load_leaf  # noqa: E402
from utils.benchmark import (BENCHMARK_VERSION, data_uids, is_panel_alias,  # noqa: E402
                             langs_for, leaf_problems)
sys.path.insert(0, SCRIPT_DIR)
from rd_common import write_json_atomic  # noqa: E402

try:                      # only the coreset bootstrap needs it; the rest is stdlib
    import numpy as np
except ImportError:
    np = None

# Least share of a panel's weight a leaf must cover before its restriction is
# reported as that panel's estimate. Below this the row is an estimate of whichever
# items happened to be scored, not of the panel.
MIN_PANEL_COVERAGE = 0.9

# Shuffle-floor MPM (utils/score_mpm_shuffle_floor.py): a distributional prosody
# metric with a token-space noise floor. Two readouts per (system, language):
#   score = d_noise/(d_ref+d_noise)*100   -- bounded but saturating
#   span  = (d_ref - d_ref_floor)/(d_noise_floor - d_ref_floor)
#           position in [0=indistinguishable-from-reference .. 1=fully scrambled];
#           this is the discriminating axis (LOWER is better).
# Results live outside the per-leaf tree in results/_mpm_shuffle_floor/, one file per
# language (clean_<lang>.tsv, every system of that language scored against one
# anchor). Joined onto clean records by (system, language).
MPM_DIR = os.path.join(ROOT, "_mpm_shuffle_floor")
MPM_GT_DIR = os.path.join(ROOT, "_mpm_ground_truth")


# ---------------------------------------------------------------------------
# Version gate. Every score file / CSV / TSV this module reads carries a
# BENCHMARK_VERSION stamp (utils/benchmark.py). Per-leaf stamps are checked with
# utils.benchmark.leaf_problems in collect_all(); the aggregate CSV/TSV loaders
# below record each row whose `benchmark_version` is not current in _STALE.
# Nothing is silently dropped: collect_all() refuses to write a snapshot while any
# input is stale, unless the leaf is listed in excluded_leaves.txt.
# ---------------------------------------------------------------------------
EXCLUDED_FILE = os.path.join(SCRIPT_DIR, "excluded_leaves.txt")
CEILING_SYSTEM = "__CEILING_split_half__"


def load_excluded(path=EXCLUDED_FILE):
    """{(model, task, lang): reason} from excluded_leaves.txt.

    One leaf per line, `<model> <task|-> <lang> <reason...>`; `-` is the task of a
    flat baseline dir (results/gt_*/<lang>/). `#` lines and blanks are ignored."""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(None, 3)
        if len(p) < 4:
            sys.exit(f"{path}:{n}: expected `<model> <task|-> <lang> <reason...>`, got {line!r}")
        out[(p[0], "" if p[1] == "-" else p[1], p[2])] = p[3]
    return out


EXCLUDED = load_excluded()


def leaf_key(system, task, lang):
    """(model, task, lang) record key of a system in a task: gt_* baselines are
    stored task-less (flat results/gt_*/<lang>/ dirs)."""
    return (system, "" if system.startswith("gt_") else task, lang)


_STALE = collections.defaultdict(set)    # (relpath, version) -> {system}


def stamp_current(path, system, version):
    """True iff `version` is the current benchmark version; else record the row."""
    if version == BENCHMARK_VERSION:
        return True
    _STALE[(os.path.relpath(path, REPO_ROOT), version)].add(system)
    return False


def stamp_problems():
    """One human-readable line per (file, stale version) seen by the row loaders."""
    return [f"{p}: benchmark_version {v if v is not None else '<missing>'} != "
            f"{BENCHMARK_VERSION} for {len(s)} system(s): {', '.join(sorted(s))}"
            for (p, v), s in sorted(_STALE.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]


def load_mpm():
    """{(system, lang): {"span": float, "score": float}} from the MPM tsvs.

    Rows whose `benchmark_version` is missing or not current are not loaded; they
    are recorded for the collect_all() refusal instead."""
    out = {}
    if not os.path.isdir(MPM_DIR):
        return out

    def add(path, lang):
        with open(path) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {k: i for i, k in enumerate(hdr)}
            if not {"system", "span", "score"} <= ix.keys():
                return
            vcol = ix.get("benchmark_version")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) <= max(ix["system"], ix["span"], ix["score"]):
                    continue
                system = p[ix["system"]]
                if leaf_key(system, "clean", lang) in EXCLUDED:
                    continue
                ver = p[vcol] if vcol is not None and vcol < len(p) else None
                if not stamp_current(path, system, ver):
                    continue
                try:
                    out[(system, lang)] = {
                        "span": float(p[ix["span"]]), "score": float(p[ix["score"]])}
                except ValueError:
                    pass

    files = sorted(f for f in os.listdir(MPM_DIR) if f.endswith(".tsv"))
    for fn in files:
        if fn.startswith("clean_"):
            add(os.path.join(MPM_DIR, fn), fn[len("clean_"):-len(".tsv")])
    if os.path.isdir(MPM_GT_DIR):
        for fn in sorted(f for f in os.listdir(MPM_GT_DIR)
                         if f.startswith("clean_") and f.endswith(".tsv")):
            add(os.path.join(MPM_GT_DIR, fn), fn[len("clean_"):-len(".tsv")])
    return out


MPM = load_mpm()


# Prosody CSVs written by scripts/rhythm_dist.py, scripts/pitch_range_dist.py and
# scripts/paired_itw.py into results/. Each file is one (task, lang); a row is one
# system. The task is the file-name prefix after <prefix>_ (see split_task_lang).
PROSODY = {}      # (model, task, lang) -> {record key: value}
_PROSODY_TASKS = ("in-the-wild", "clean")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def split_task_lang(stem):
    """`in-the-wild_en` -> (`in-the-wild`, `en`); None if not a task stem."""
    for task in _PROSODY_TASKS:
        if stem.startswith(task + "_") and len(stem) > len(task) + 1:
            return task, stem[len(task) + 1:]
    return None


def load_prosody(prefix, columns, fixed_task=None):
    """Fold results/<prefix>_<task>_<lang>.csv (or <prefix>_<lang>.csv when
    `fixed_task` is given) into PROSODY. `columns` maps record key -> CSV column.
    The split-half ceiling row and whisper_* diagnostics are skipped."""
    if not os.path.isdir(ROOT):
        return
    for fn in sorted(os.listdir(ROOT)):
        if not (fn.startswith(prefix + "_") and fn.endswith(".csv")):
            continue
        stem = fn[len(prefix) + 1:-len(".csv")]
        if fixed_task:
            task, lang = fixed_task, stem
        else:
            tl = split_task_lang(stem)
            if tl is None:
                continue
            task, lang = tl
        path = os.path.join(ROOT, fn)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                system = row.get("system") or ""
                if not system or system == CEILING_SYSTEM or system.startswith("whisper_"):
                    continue
                key = leaf_key(system, task, lang)
                if key in EXCLUDED:
                    continue
                if not stamp_current(path, system, row.get("benchmark_version")):
                    continue
                vals = {k: _num(row.get(col)) for k, col in columns.items()}
                PROSODY.setdefault(key, {}).update(
                    {k: v for k, v in vals.items() if v is not None})


# rhythm: pause/stall factor (burstiness) is the headline; pace factor is shown beside it.
load_prosody("rhythm", {"rhythm_x": "burstiness_x_ceiling",
                        "rhythm_rate_x": "rate_x_ceiling"})
load_prosody("pitch_range", {"pitch_range_x": "x_ceiling"})
load_prosody("paired_itw", {"paired_dur": "dur_abs_med",
                            "paired_pause": "pause_abs_mean",
                            "paired_range": "range_abs_med"},
             fixed_task="in-the-wild")


REC_KEYS = ("wer", "wer_ci", "tail50", "spk", "spk_xutt", "msr_utmos", "msr_utmos_ci",
            "mpm_span", "rhythm_x", "rhythm_rate_x", "pitch_range_x",
            "paired_dur", "paired_pause", "paired_range", "n")


def new_rec(model, task, setname):
    """An all-None record in the snapshot's schema."""
    rec = {"model": model, "task": task, "set": setname}
    rec.update(dict.fromkeys(REC_KEYS))
    return rec


# ---------------------------------------------------------------------------
# Coreset panels (scripts/coreset/) -- clean_mini, clean_delta, and the
# VoxBlink twins. A panel leaf must NOT be summarised by the `.wer` footer: that
# footer is an unweighted mean over the sampled items, whereas a panel's estimate
# is the weighted mean whose weights carry each selected item's cluster mass
#   S = (1/|L|) * sum_l sum_{j in A_l} w_j y_j ,   sum_j w_j = 1 within a language
# (see docs/coreset.md). Weights are within ~10% of uniform for the Mini panels, so
# the unweighted number looks plausible while being a different estimator. Panel
# leaves are therefore re-aggregated here from the per-uid score files.
# ---------------------------------------------------------------------------
def load_panels():
    """{task: {kind, source_task, selector, k_actual, w: {lang: {uid: weight}}}}."""
    out = {}
    if not os.path.isdir(DATA):
        return out
    for task in sorted(os.listdir(DATA)):
        mpath = os.path.join(DATA, task, "panel.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            continue
        out[task] = {
            "kind": man.get("kind"),
            "source_task": man.get("source_task"),
            "selector": man.get("selector"),
            "k_actual": man.get("k_actual"),
            "seed": man.get("seed"),
            "w": {lg: dict(pairs) for lg, pairs in (man.get("langs") or {}).items()},
        }
    return out


PANELS = load_panels()


def wmean(y, w):
    tot = sum(w)
    return (sum(a * b for a, b in zip(y, w)) / tot) if tot else None


def wboot_ci(y, w, boot=CI_BOOT, seed=CI_SEED):
    """95% CI for the weighted mean, resampling items with replacement and
    recomputing sum(w*y)/sum(w) -- the same scheme as scripts/coreset/score_panel.py
    so a board cell and the panel scorer agree. Returns (None, None) without numpy."""
    if np is None or len(y) < 2:
        return None, None
    ya, wa = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    rng = np.random.default_rng(seed)
    j = rng.integers(0, ya.size, size=(boot, ya.size))
    draws = (wa[j] * ya[j]).sum(1) / wa[j].sum(1)
    return round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)


def macro_boot_ci(pieces, boot=CI_BOOT, seed=CI_SEED):
    """95% CI for the macro estimate: resample items WITHIN each language, take that
    language's weighted mean, then average across languages -- score_panel.py's
    boot_ci(). Resampling within language is what makes it a CI for the macro: the
    languages are the panel's fixed strata, not a sample from a population of them."""
    if np is None or not pieces or all(len(y) < 2 for y, _w in pieces):
        return None, None
    rng = np.random.default_rng(seed)
    acc = np.empty((len(pieces), boot))
    for i, (y, w) in enumerate(pieces):
        ya, wa = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
        j = rng.integers(0, ya.size, size=(boot, ya.size))
        acc[i] = (wa[j] * ya[j]).sum(1) / wa[j].sum(1)
    draws = acc.mean(0)
    return round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)


def collect_panel_leaf(model, task, setname, leaf_dir, panel):
    """Weighted re-aggregation of one panel leaf, in the same record shape as
    collect_leaf(). Returns None when the language is not in the panel or nothing
    on disk overlaps it."""
    w_lang = panel["w"].get(setname)
    if not w_lang:
        return None
    leaf = load_leaf(leaf_dir)
    # The GT baseline's prompt-vs-real-target score IS the held-out ceiling.
    spk_xutt = leaf["spk"] if model == "gt_in-the-wild" else leaf["spk_xutt"]
    rec = new_rec(model, task, setname)
    rec["panel"] = {"kind": panel["kind"], "source_task": panel["source_task"],
                    "selector": panel["selector"], "k": len(w_lang),
                    "coverage": 0.0}

    uids = [u for u in sorted(w_lang) if u in leaf["wer"]]
    total_w = sum(w_lang.values())
    if total_w:
        rec["panel"]["coverage"] = round(sum(w_lang[u] for u in uids) / total_w, 4)
    # `_pieces` are the raw (y, w) arrays behind each cell, kept on the record so the
    # ALL row can macro-average them without re-reading every leaf. Stripped before
    # the snapshot is written -- they are build state, not payload.
    pieces = {}
    if uids:
        w = [w_lang[u] for u in uids]
        raw = [leaf["wer"][u] * 100.0 for u in uids]
        clipped = [leaf["wer_clip"][u] for u in uids]
        # WER_L50 is a conditional mean over a subset-dependent filter, not a plain
        # weighted mean, so it is reported (weights renormalised within the kept
        # items) but is approximate on a panel -- see docs/coreset.md.
        keep = [i for i, u in enumerate(uids) if leaf["wer"][u] <= L50_THRESHOLD]
        l50 = (wmean([raw[i] for i in keep], [w[i] for i in keep])
               if keep else wmean(raw, w))
        rec["wer"] = [round(wmean(raw, w), 3), round(l50, 3), round(wmean(clipped, w), 3)]
        wlo, whi = wboot_ci(raw, w)
        clo, chi = wboot_ci(clipped, w)
        if wlo is not None:
            rec["wer_ci"] = [wlo, whi, clo, chi]
        # Tail rate: share of (weighted) utterances with WER above 50%, the same
        # threshold average_wer.py's WER_TAIL50 footer uses.
        tail = [100.0 if leaf["wer"][u] > L50_THRESHOLD else 0.0 for u in uids]
        rec["tail50"] = round(wmean(tail, w), 3)
        rec["n"] = len(uids)
        pieces["wer"] = (raw, w)
        pieces["wer_l50"] = ([raw[i] for i in keep], [w[i] for i in keep]) if keep else (raw, w)
        pieces["wer_clip"] = (clipped, w)
        pieces["tail50"] = (tail, w)

    for key, scores in (("spk", leaf["spk"]), ("spk_xutt", spk_xutt)):
        su = [u for u in sorted(w_lang) if u in scores]
        if su:
            sw = [w_lang[u] for u in su]
            sy = [scores[u] for u in su]
            rec[key] = round(wmean(sy, sw), 3)
            pieces[key] = (sy, sw)

    mu = [u for u in sorted(w_lang) if u in leaf["msr"]]
    if mu:
        mw = [w_lang[u] for u in mu]
        my = [leaf["msr"][u] for u in mu]
        rec["msr_utmos"] = round(wmean(my, mw), 4)
        lo, hi = wboot_ci(my, mw)
        if lo is not None:
            rec["msr_utmos_ci"] = [lo, hi]
        if rec["n"] is None:
            rec["n"] = len(mu)
        pieces["msr_utmos"] = (my, mw)

    rec["_pieces"] = pieces
    return rec if rec["n"] is not None else None


# The footers grew (CI, N_IMPUTED, WER_TAIL50, BENCHMARK_VERSION, ASR; the speaker
# files carry four trailing `#` lines), so every footer parser reads a 4096-byte tail.
TAIL_BYTES = 4096
WER_RE = re.compile(r"WER:\s*([\-\d\.eN]+(?:nan|inf)?|nan|inf|[\d\.\-eE]+),\s*WER_L50:\s*([\d\.\-eEnan]+|nan),\s*WER_CLIP:\s*([\d\.\-eEnan]+|nan)")
# 95% CI footer line emitted by average_wer.py (optional; absent on pre-CI results).
WER_CI_RE = re.compile(r"WER_CI95:\s*\[([\d\.\-eEnan]+),\s*([\d\.\-eEnan]+)\],\s*WER_CLIP_CI95:\s*\[([\d\.\-eEnan]+),\s*([\d\.\-eEnan]+)\]")
MSR_CI_RE = re.compile(r"CI95\s+([\d\.\-eEnan]+)\s+([\d\.\-eEnan]+)")


def read_tail(path, nbytes=TAIL_BYTES):
    """Last `nbytes` of a file as text, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-nbytes, os.SEEK_END)
            except OSError:
                f.seek(0)
            return f.read().decode(errors="replace")
    except Exception:
        return None


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_wer(path):
    """Last `WER:` footer line of a .wer file; return (wer, l50, clip) or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("WER:"):
            m = WER_RE.search(line)
            if m:
                return _float(m.group(1)), _float(m.group(2)), _float(m.group(3))
    return None


def parse_wer_ci(path):
    """(wer_lo, wer_hi, clip_lo, clip_hi) from the WER_CI95 footer, or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("WER_CI95:"):
            m = WER_CI_RE.search(line)
            if m:
                return tuple(_float(m.group(i)) for i in (1, 2, 3, 4))
    return None


def parse_tail50(path):
    """Share (%) of utterances with WER > 50%, from the `WER_TAIL50:` footer, or None."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("WER_TAIL50:"):
            v = _float(line.split(":", 1)[1].strip())
            return v if v == v else None
    return None


def parse_msr_ci(path):
    """(lo, hi) from a msr_utmos_ci.txt line, or None."""
    try:
        with open(path) as f:
            m = MSR_CI_RE.search(f.read())
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception:
        return None
    return None


def parse_spk(path):
    """Mean hyp score (cosine x100) from the `avg` line of a speaker-score file."""
    tail = read_tail(path)
    if tail is None:
        return None
    for line in reversed(tail.splitlines()):
        if line.startswith("avg"):
            parts = line.split()
            try:
                return float(parts[2])
            except (IndexError, ValueError):
                return None
    return None


def parse_msr_utmos(path):
    try:
        with open(path) as f:
            v = float(f.read().strip())
    except Exception:
        return None
    # A 0.0 mean is the scorer's historical crash stub (MOS scale is ~1-5, a real
    # mean cannot be 0) -- treat as unscored rather than folding it into averages.
    return v if v > 0.0 else None

def count_n(wer_raw_path):
    try:
        n = 0
        with open(wer_raw_path) as f:
            for _ in f: n += 1
        return n
    except Exception:
        return None

# A full-set leaf counts only when its system generated (nearly) every item of the
# language: same bar as the scorers' SCORE_MIN_COVERAGE.
MIN_GEN_COVERAGE = 0.98


def generation_coverage(task, lang, leaf_dir):
    """(generated items, items) of a full-set leaf, or None when there is no data
    definition to compare against (unknown task / language)."""
    try:
        uids = set(data_uids(task, lang))
    except OSError:
        return None
    wav_dir = os.path.join(leaf_dir, "wavs")
    have = ({f[:-4] for f in os.listdir(wav_dir) if f.endswith(".wav")}
            if os.path.isdir(wav_dir) else set())
    return len(have & uids), len(uids)

def collect_leaf(model, task, setname, leaf_dir):
    wer_path = os.path.join(leaf_dir, "wav_res_ref_text.wer")
    if not os.path.isfile(wer_path):
        return None
    rec = new_rec(model, task, setname)
    # Shuffle-floor MPM is a FLEURS-R-only (Clean set) distributional metric, keyed
    # (system, lang). Only `span` (the discriminating axis) is surfaced; `score`
    # saturates, so it is intentionally not carried into the snapshot. The clean
    # restriction is not an oversight: MPM's anchors collapse on in-the-wild audio
    # and no front-end or anchor fix recovers them -- see docs/mpm_itw_limits.md.
    mpm_model = model if task == "clean" or (not task and model == "gt_clean") else None
    if mpm_model:
        m = MPM.get((mpm_model, setname))
        if m:
            rec["mpm_span"] = m["span"]
    rec["wer"] = parse_wer(wer_path)
    rec["wer_ci"] = parse_wer_ci(wer_path)
    rec["tail50"] = parse_tail50(wer_path)
    sp = os.path.join(leaf_dir, "spk_simi_scores.txt")
    if os.path.isfile(sp): rec["spk"] = parse_spk(sp)
    if model == "gt_in-the-wild" and not task:
        # The GT leaf scores the prompt against the real target recording: that
        # score IS the held-out ceiling, so there is no separate xutt file.
        rec["spk_xutt"] = rec["spk"]
    else:
        xp = os.path.join(leaf_dir, "spk_simi_scores.xutt.txt")
        if os.path.isfile(xp): rec["spk_xutt"] = parse_spk(xp)
    mup = os.path.join(leaf_dir, "msr_utmos_mean.txt")
    if os.path.isfile(mup): rec["msr_utmos"] = parse_msr_utmos(mup)
    mci = os.path.join(leaf_dir, "msr_utmos_ci.txt")
    if os.path.isfile(mci): rec["msr_utmos_ci"] = parse_msr_ci(mci)
    rec.update(PROSODY.get((model, task, setname), {}))
    rp = os.path.join(leaf_dir, "wav_res_ref_text.wer.raw")
    if os.path.isfile(rp): rec["n"] = count_n(rp)
    return rec

# Walk: results/<model>/<task>/<set>/  OR results/<model>/<set>/
def collect_all():
    records = []
    if not os.path.isdir(ROOT):
        return records
    problems = []          # inputs that are not current ZTTS2-Eval <BENCHMARK_VERSION>
    audited = set()
    incomplete = []        # full-set leaves whose generation is missing items

    def audit(leaf_dir, lang):
        if leaf_dir in audited:
            return
        audited.add(leaf_dir)
        # leaf_problems() messages already name the offending file
        problems.extend(leaf_problems(leaf_dir, lang))

    for model in sorted(os.listdir(ROOT)):
        # whisper_* are ASR-comparison diagnostics (read directly by the reliability
        # and coreset tools); `_*` are aggregate caches, not systems.
        if model.startswith(("whisper_", "_")):
            continue
        mdir = os.path.join(ROOT, model)
        if not os.path.isdir(mdir): continue
        for sub in sorted(os.listdir(mdir)):
            sdir = os.path.join(mdir, sub)
            if not os.path.isdir(sdir): continue
            if is_panel_alias(mdir, sub):
                continue   # browse-only alias of the full set; the backfill below scores it
            if (model, "", sub) in EXCLUDED:
                continue
            rec = collect_leaf(model, "", sub, sdir)
            if rec is not None:
                records.append(rec)
                audit(sdir, sub)
                continue
            for setname in sorted(os.listdir(sdir)):
                tdir = os.path.join(sdir, setname)
                if not os.path.isdir(tdir): continue
                if (model, sub, setname) in EXCLUDED:
                    continue
                panel = PANELS.get(sub)
                if panel is not None:
                    rec = collect_panel_leaf(model, sub, setname, tdir, panel)
                else:
                    gen = generation_coverage(sub, setname, tdir)
                    if gen is not None and gen[0] < MIN_GEN_COVERAGE * gen[1]:
                        incomplete.append(f"{model}/{sub}/{setname} ({gen[0]}/{gen[1]})")
                        continue
                    rec = collect_leaf(model, sub, setname, tdir)
                if rec is not None:
                    records.append(rec)
                    audit(tdir, setname)
    if incomplete:
        # The scorers only see the wavs that exist, so a leaf whose generation stopped
        # early would otherwise report full coverage over a handful of utterances.
        print(f"leaves dropped, generation covers < {MIN_GEN_COVERAGE:.0%} of the "
              f"language's items ({len(incomplete)}): {', '.join(incomplete)}")

    # Backfill every panel row that can be derived from the full source task.
    #
    # A panel is a weighted SUBSET of the source task's items, so for any system
    # already scored on the full task the panel estimate is obtained by restricting
    # that leaf -- the panel's uids are the same texts and the same prompts, scored
    # from the same generations, which is exactly how the offline validation
    # (scripts/coreset/validate.py) computed them. Without this, a panel board shows
    # only the handful of systems someone re-ran on the panel task, which is the
    # least useful version of a board whose whole point is cheap comparison.
    #
    # Two things this is NOT: it is not an independent check that the panel works
    # (for the systems the panel was selected on, these are the very numbers it was
    # selected from), and it does not replace a genuine panel run -- rows produced by
    # actually generating on the panel task are found by the walk above and kept in
    # preference to a backfill, so the board never overwrites a real measurement.
    have = {(r["model"], r["task"], r["set"]) for r in records}
    thin = []            # dropped for thin coverage -- reported, never silent
    for ptask, panel in sorted(PANELS.items()):
        src = panel.get("source_task")
        if not src:
            continue
        for model in sorted(os.listdir(ROOT)):
            if model.startswith(("whisper_", "_")):
                continue
            # nested results/<model>/<src>/<lang>/, or the baselines' flat
            # results/gt_<src>/<lang>/ (their task lives in the dir name)
            base = os.path.join(ROOT, model, src)
            flat = model == f"gt_{src}"
            if not flat and not os.path.isdir(base):
                continue
            for setname in sorted(panel["w"]):
                if (model, ptask, setname) in have:
                    continue
                if (model, "" if flat else src, setname) in EXCLUDED:
                    continue
                ldir = os.path.join(ROOT, model, setname) if flat \
                    else os.path.join(base, setname)
                if not os.path.isdir(ldir):
                    continue
                rec = collect_panel_leaf(model, ptask, setname, ldir, panel)
                if rec is None:
                    continue
                # A partial leaf renormalises the weights over whatever it covers,
                # which silently estimates a different (self-selected) population --
                # so carry the covered weight and drop rows that cover too little.
                if rec["panel"]["coverage"] < MIN_PANEL_COVERAGE:
                    thin.append(f"{model}/{ptask}/{setname} "
                                f"({rec['panel']['coverage']:.0%})")
                    continue
                rec["panel"]["backfilled"] = True
                records.append(rec)
                audit(ldir, setname)
    if thin:
        print(f"panel rows dropped, leaf covers < {MIN_PANEL_COVERAGE:.0%} of the "
              f"panel weight ({len(thin)}): {', '.join(sorted(thin))}")

    problems.extend(stamp_problems())
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(f"build_results: {len(problems)} input(s) are not ZTTS2-Eval "
                 f"{BENCHMARK_VERSION}; rescore (scripts/rescore_all.sh, "
                 "slurm/submit_metrics.sh) or list the leaf in "
                 "scripts/results_doc/excluded_leaves.txt")

    records.extend(panel_aggregates(records))
    records.extend(full_aggregates(records))
    for r in records:
        r.pop("_pieces", None)
    return records


# A panel's estimate is the macro-average over its languages -- the number
# scripts/coreset/score_panel.py prints and the only one the panel was designed and
# validated to deliver. Split across per-language cells it is unreadable at best and
# misleading at worst: a Delta panel puts k=200 over 35 languages, so a cell rests on
# 3-9 items and systems routinely tie on it. So each (system, panel) also gets one
# row carrying the aggregate, under the reserved set name ALL_SET.
ALL_SET = "ALL"


def panel_aggregates(records):
    """One macro-averaged row per (model, panel task), set=ALL_SET.

    Mirrors score_panel.py: each language contributes its weighted mean, those are
    averaged unweighted (every language counts once, which is the definition of the
    panel estimator), and the CI resamples items within language before averaging.

    Only systems covering EVERY language the panel scored for some system are
    aggregated: a macro-average over a different language subset is a different
    quantity, and putting it in the same column as the full one invites exactly the
    comparison it cannot support. Partial systems keep their per-language rows and
    are reported on stderr.
    """
    by_task = collections.defaultdict(list)
    for r in records:
        if r.get("panel") and r["set"] != ALL_SET:
            by_task[r["task"]].append(r)

    out, partial = [], []
    for task, rs in sorted(by_task.items()):
        full = {r["set"] for r in rs}
        by_model = collections.defaultdict(list)
        for r in rs:
            by_model[r["model"]].append(r)
        for model, mrs in sorted(by_model.items()):
            have = {r["set"] for r in mrs}
            if have != full:
                partial.append(f"{model}/{task} ({len(have)}/{len(full)} langs)")
                continue
            p0 = mrs[0]["panel"]
            rec = new_rec(model, task, ALL_SET)
            rec["n"] = sum(r["n"] or 0 for r in mrs)
            rec["panel"] = {
                "kind": p0["kind"], "source_task": p0["source_task"],
                "selector": p0["selector"],
                "k": sum(r["panel"]["k"] for r in mrs),
                "coverage": round(sum(r["panel"]["coverage"] for r in mrs) / len(mrs), 4),
                "langs": len(mrs),
                # `run` only if every language was generated on the panel task
                "backfilled": any(r["panel"].get("backfilled") for r in mrs),
                "aggregate": True,
            }

            def macro(key, nd):
                pieces = [r["_pieces"][key] for r in mrs if key in r["_pieces"]]
                if len(pieces) != len(mrs):     # a metric absent for some language
                    return None, (None, None)   # would macro over a different subset
                per_lang = [wmean(y, w) for y, w in pieces]
                return round(sum(per_lang) / len(per_lang), nd), macro_boot_ci(pieces)

            raw, (wlo, whi) = macro("wer", 3)
            l50, _ = macro("wer_l50", 3)
            clipv, (clo, chi) = macro("wer_clip", 3)
            if raw is not None and clipv is not None:
                rec["wer"] = [raw, l50 if l50 is not None else raw, clipv]
                if wlo is not None and clo is not None:
                    rec["wer_ci"] = [wlo, whi, clo, chi]
            rec["tail50"] = macro("tail50", 3)[0]
            rec["spk"] = macro("spk", 3)[0]
            rec["spk_xutt"] = macro("spk_xutt", 3)[0]
            msr, (mlo, mhi) = macro("msr_utmos", 4)
            rec["msr_utmos"] = msr
            if mlo is not None:
                rec["msr_utmos_ci"] = [mlo, mhi]
            if rec["wer"] or rec["msr_utmos"] is not None:
                out.append(rec)
    if partial:
        print(f"panel ALL rows skipped, system misses languages ({len(partial)}): "
              f"{', '.join(sorted(partial))}")
    return out


# The two full sets get an ALL row too: the unweighted macro-average over the set's
# frozen language basket (utils.benchmark.langs_for), the number the headline tables of
# RESULTS_CURRENT.md show (rd_common.macro) -- so the explorer and the document cannot
# disagree. Same rule as the panels: a metric is defined only when the system has it in
# EVERY basket language, because a mean over a different language subset is a different
# quantity. Unlike a panel, a system missing languages still gets its ALL row, with every
# metric None and its coverage in `macro`, so the board says why rather than dropping it.
FULL_TASKS = ("clean", "in-the-wild")
_MACRO_KEYS = ("tail50", "spk", "spk_xutt", "msr_utmos", "mpm_span", "rhythm_x",
               "rhythm_rate_x", "pitch_range_x", "paired_dur", "paired_pause", "paired_range")


def full_aggregates(records):
    """One macro row per (system, full task), set=ALL_SET; baselines (gt_<task>) keep
    their flat task "" like their per-language rows."""
    ok = lambda v: v is not None and v == v      # NaN counts as missing, as in rd_common
    out = []
    for task in FULL_TASKS:
        basket = langs_for(task)
        by_model = collections.defaultdict(dict)
        for r in records:
            if r["set"] == ALL_SET or r.get("panel"):
                continue
            if r["task"] == task or (r["task"] == "" and r["model"] == f"gt_{task}"):
                by_model[r["model"]][r["set"]] = r
        for model, rows in sorted(by_model.items()):
            have = [l for l in basket if l in rows]
            if not have:
                continue
            rec = new_rec(model, rows[have[0]]["task"], ALL_SET)
            rec["n"] = sum(rows[l]["n"] or 0 for l in have)
            rec["macro"] = {"langs": len(have), "of": len(basket)}
            if len(have) == len(basket):
                for key in _MACRO_KEYS:
                    vals = [rows[l][key] for l in basket]
                    if all(ok(v) for v in vals):
                        rec[key] = round(sum(vals) / len(vals), 4)
                wers = [rows[l]["wer"] for l in basket]
                if all(w and all(ok(c) for c in w) for w in wers):
                    rec["wer"] = [round(sum(w[i] for w in wers) / len(wers), 3) for i in range(3)]
            out.append(rec)
    return out


def jsonable(x):
    """NaN/Infinity -> None, recursively.

    `json.dump` happily writes the bare tokens `NaN` / `Infinity`, which are *not*
    valid JSON: `JSON.parse` rejects them. The explorer survived that only because
    its primary loader is JSONP, where those tokens are legal JavaScript literals —
    its plain-`fetch` fallback (and any other consumer) throws. A missing CI is a
    real state here (a leaf with n=0 writes `CI95 nan nan`), so it is emitted as
    null rather than repaired."""
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if isinstance(x, (list, tuple)):  # parse_wer/parse_wer_ci hand back tuples
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    return x


def main():
    if not os.path.isdir(ROOT):
        print(f"results directory not found; writing empty snapshot to {OUT_JSON}")
    records = jsonable(collect_all())

    write_json_atomic(OUT_JSON, records, indent=2, allow_nan=False)

    # Panel provenance, without the per-uid weights: the explorer only needs to know
    # which tasks are panels and how to caption them. Shipped alongside the snapshot so
    # a clone that has no data/ tree still labels the panel boards correctly.
    write_json_atomic(os.path.join(CACHE, "panels.json"),
                      {t: dict({k: v for k, v in p.items() if k != "w"}, langs=len(p["w"]))
                       for t, p in PANELS.items()}, indent=2)

    print(f"{len(records)} records written to {OUT_JSON}")
    print("Models:", sorted({r['model'] for r in records}))
    print("Tasks:", sorted({r['task'] for r in records}))
    if PANELS:
        n_panel = sum(1 for r in records if r.get("panel"))
        print(f"Panels: {sorted(PANELS)} ({n_panel} weighted rows)"
              + ("" if np is not None else "  [numpy absent -- panel CIs omitted]"))


if __name__ == "__main__":
    main()
