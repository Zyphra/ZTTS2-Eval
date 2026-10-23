"""Ridgeline view of the retained TTSDS2 Pitch distributions.

The raw per-file mean-F0 values cached by TTSDS2 are plotted for each system
against the real-speech reference. Rows are sorted by Wasserstein-1 distance,
so this view exposes the distribution behind the normalized TTSDS score.

The default run loads a frozen snapshot from
``assets/plots/ttsds_pitch_ridgeline_data.npz``. Pass ``--refresh`` in the
TTSDS venv to rebuild that snapshot from the live cache and wav directories.
"""
import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde, wasserstein_distance


REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache/ttsds/benchmarks"
FROZEN = REPO / "assets/plots/ttsds_pitch_ridgeline_data.npz"

# Open / local models: a single results dir each (also the TTSDS2 dataset name).
# Subject = the released Zonos2 model, accurate vs expressive mode.
OPEN_MODELS = [
    ("omni_fish-s2-pro",     "fish-s2-pro"),
    ("zonos2_accurate",      "zonos2 (accurate)"),
    ("zonos2_expressive",    "zonos2 (expressive)"),
    ("omni_qwen3-tts-1.7b",  "qwen3-tts-1.7b"),
    ("omni_voxcpm2",         "voxcpm2"),
]

# Closed / API models: prefer the per-utterance voice-clone run (apples-to-apples
# zero-shot vs the reference speaker); fall back to the preset system-voice run
# wherever no clone generation exists for a given test set. The chosen dir name
# is also the TTSDS2 dataset name used when it was scored, so the cache hash
# matches. gemini has no clone path, so it is preset-only.
# NOTE: these runs are historical — the dirs may be absent from the current
# results/ tree, in which case --refresh warns and omits them (the frozen
# snapshot keeps whatever was frozen when they existed).
CLOSED_SOURCES = {
    "gemini-3.1-flash":   {"clone": None,                   "preset": "gemini_3.1-flash"},
    "cartesia-sonic-3.5": {"clone": "cartesia_sonic-3.5_clone", "preset": "cartesia_sonic-3.5"},
    "elevenlabs-v3":      {"clone": "elevenlabs_v3_clone",  "preset": "elevenlabs_v3"},
    "inworld":            {"clone": "inworld_2_clone",      "preset": "inworld_2"},
}
CLOSED_LABELS = set(CLOSED_SOURCES)
# Display order across all rows (open first, then closed), used for colour lookup.
MODEL_ORDER = [s for _, s in OPEN_MODELS] + list(CLOSED_SOURCES)


def resolve_source(set_key, short):
    """Return (ttsds_dataset_name, mode) for a model in a given set, or None.

    Open models map to their single dir. Closed models prefer their clone dir
    when it actually has wavs for this set, else fall back to preset.
    """
    if short not in CLOSED_LABELS:
        for d, s in OPEN_MODELS:
            if s == short:
                return d, "open"
        return None
    src = CLOSED_SOURCES[short]
    clone = src["clone"]
    if clone and (REPO / f"results/{clone}/{set_key}/en/wavs").is_dir() \
            and any((REPO / f"results/{clone}/{set_key}/en/wavs").glob("*.wav")):
        return clone, "clone"
    return src["preset"], "preset"

SETS = [
    ("zero_shot",          "prompt_audio_en",            "zero_shot/en — read-aloud prompts"),
    ("zero_shot_voxblink", "ref_zero_shot_voxblink_en",  "zero_shot_voxblink/en — in-the-wild"),
]

BENCHES = [
    ("Pitch", "mean F0 (Hz)", (50, 350)),
]

# --- Zyphra dark-mode tokens ---------------------------------------------
BG, PANEL, GRID = "#0A0A0A", "#171717", "#262626"
T1, T2, MUTED   = "#FAFAFA", "#D4D4D4", "#737373"
ORANGE          = "#F97316"
GOLD = "#FBBF24"
FAMILY = {                       # subjects vivid (orange/gold); competitors soft
    "zonos2 (accurate)":   ORANGE,  # subject — released model, accurate mode
    "zonos2 (expressive)": GOLD,    # subject — expressive mode
    "qwen3-tts-1.7b":     "#A693BD",   # Qwen purple
    "fish-s2-pro":        "#7AAEC0",   # soft blue
    "voxcpm2":            "#A8B07A",   # soft green
    "gemini-3.1-flash":   "#5A7E9C",   # muted blue-gray
    "cartesia-sonic-3.5": "#9E8BB8",   # muted lavender
    "elevenlabs-v3":      "#B87963",   # warm terracotta
    "inworld":            "#737373",   # neutral gray
}
SUBJECT = {"zonos2 (accurate)", "zonos2 (expressive)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": T2})


def hexv(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


BGV = hexv(BG)


def nice_ticks(lo, hi, n=7):
    raw = (hi - lo) / (n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    start = math.ceil(lo / step) * step
    return np.arange(start, hi + step * 1e-6, step)


def load_frozen():
    """Load the committed snapshot -> {bench: (rows, modes)}."""
    z = np.load(FROZEN, allow_pickle=True)
    meta = z["meta"]
    data = {bench: ({}, {}) for bench, _, _ in BENCHES}
    for i, (bench, set_key, label) in enumerate(meta):
        data[bench][0][(set_key, label)] = z[f"a{i}"]
    for bench, set_key, short, mode in z["modes"]:
        data[bench][1][(set_key, short)] = mode
    return data


def refresh_from_cache():
    """Rebuild the per-file Pitch arrays from the live TTSDS2 cache."""
    from ttsds.util.cache import hash_md5
    from ttsds.util.dataset import DirectoryDataset

    def ds_hash(root_dir, name, sample_n=None):
        ds = DirectoryDataset(str(Path(root_dir).resolve()), name=name, sample_rate=22050)
        if sample_n:
            ds.sample(sample_n)
        return hash_md5(ds)

    def find_npy(bench_dir, prefix):
        matches = [m for m in sorted(bench_dir.glob(f"{prefix}_*.npy"))
                   if "_mu" not in m.stem and "_sig" not in m.stem]
        arrs = [(m, np.load(m, allow_pickle=True)) for m in matches]
        arrs = [(m, a) for m, a in arrs if a.ndim == 1]
        return max(arrs, key=lambda x: x[1].size)[1] if arrs else None

    def collect(bench_name):
        bd = CACHE / bench_name
        rows = {}   # (set_key, label) -> array
        modes = {}  # (set_key, short)  -> "open" | "clone" | "preset"
        for set_key, ref_label, _ in SETS:
            ref_root = REPO / f"data/{set_key}/en/waveform"
            arr = find_npy(bd, ds_hash(ref_root, ref_label))
            if arr is None:
                arr = find_npy(bd, ds_hash(ref_root, ref_label, sample_n=100))
            if arr is not None:
                rows[(set_key, "reference")] = arr
            for short in MODEL_ORDER:
                src = resolve_source(set_key, short)
                if src is None:
                    continue
                model_dir, mode = src
                synth_root = REPO / f"results/{model_dir}/{set_key}/en/wavs"
                if not synth_root.is_dir():
                    print(f"[refresh] {set_key}/{short}: {synth_root} absent — "
                          "omitted from the snapshot", file=sys.stderr)
                    continue
                arr = find_npy(bd, ds_hash(synth_root, model_dir))
                if arr is not None:
                    rows[(set_key, short)] = arr
                    modes[(set_key, short)] = mode
                else:
                    print(f"[refresh] {set_key}/{short}: no cached Pitch arrays "
                          f"for {model_dir} — omitted (score it with "
                          "scripts/run_ttsds_pitch.sh first)", file=sys.stderr)
        return rows, modes

    data = {bench: collect(bench) for bench, _, _ in BENCHES}

    # Freeze: a meta object-array of (bench, set_key, label) aligned with a0,
    # a1, ... payload arrays, plus a modes object-array, so the loader needs
    # no ttsds.
    meta, payload, modes_meta = [], {}, []
    i = 0
    for bench, _, _ in BENCHES:
        rows, modes = data[bench]
        for (set_key, label), arr in rows.items():
            meta.append((bench, set_key, label))
            payload[f"a{i}"] = np.asarray(arr, dtype=np.float64)
            i += 1
        for (set_key, short), mode in modes.items():
            modes_meta.append((bench, set_key, short, mode))
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FROZEN, meta=np.array(meta, dtype=object),
                        modes=np.array(modes_meta, dtype=object), **payload)
    print(f"froze {i} distributions -> {FROZEN}")
    return data


def kde_curve(x, xs):
    x = x[np.isfinite(x)]
    if x.size == 0 or np.allclose(x, x[0]):
        return None
    try:
        return gaussian_kde(x)(xs)
    except Exception:
        return None


STEP, AMP = 0.55, 1.0


def ridgeline_ax(ax, rows, modes, set_key, xrange, xlabel, title):
    xs = np.linspace(xrange[0], xrange[1], 600)
    ax.set_facecolor(BG)

    ref = rows.get((set_key, "reference"))
    ref_y = kde_curve(ref, xs) if ref is not None else None

    present = [short for short in MODEL_ORDER if (set_key, short) in rows]
    curves = {m: kde_curve(rows[(set_key, m)], xs) for m in present}
    curves = {m: y for m, y in curves.items() if y is not None}
    present = [m for m in present if m in curves]
    if ref is not None:
        present.sort(key=lambda m: wasserstein_distance(ref, rows[(set_key, m)]))

    norm = max([y.max() for y in curves.values()]
               + ([ref_y.max()] if ref_y is not None else []) + [1e-9])

    ytrans = ax.get_yaxis_transform()  # x in axes-frac, y in data coords
    for i, name in enumerate(present):
        base = -i * STEP
        y = curves[name] / norm * AMP
        z = (i + 1) * 10
        subj = name in SUBJECT
        c = hexv(FAMILY[name])
        fill_mix = 0.55 if subj else 0.30           # subject saturated, rest soft
        cf = tuple(fill_mix * c + (1 - fill_mix) * BGV)
        ax.fill_between(xs, base, base + y, color=cf, lw=0, zorder=z)
        ax.plot(xs, base + y, color=BG, lw=3.5, zorder=z + 1)          # dark halo
        ax.plot(xs, base + y, color=FAMILY[name], lw=2.0 if subj else 1.2,
                zorder=z + 2)
        if ref_y is not None:                                         # ref outline
            ax.plot(xs, base + ref_y / norm * AMP, color=T1, lw=0.9,
                    ls=(0, (4, 2.5)), alpha=0.75, zorder=z + 3)
        mode = modes.get((set_key, name))
        label = name
        if mode in ("clone", "preset"):                # tag closed-model source
            label = f"{name}  ·{mode}"
        ax.text(-0.02, base + 0.02, label, transform=ytrans, ha="right", va="bottom",
                fontsize=9.5, zorder=99, clip_on=False,
                color=FAMILY[name] if subj else T2,
                fontweight="bold" if subj else "normal")
        if ref is not None:
            w1 = wasserstein_distance(ref, rows[(set_key, name)])
            ax.text(1.02, base + 0.02, f"{w1:.1f}", transform=ytrans, ha="left",
                    va="bottom", fontsize=10, zorder=99, clip_on=False,
                    color=FAMILY[name] if subj else T1, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    n = len(present)
    if ref is not None:
        ax.text(1.02, AMP * 0.55, "W1 → ref", transform=ytrans, ha="left",
                fontsize=8.5, color=MUTED, clip_on=False)
    ax.set_xlim(*xrange)
    ax.set_ylim(-(max(n, 1) - 1) * STEP - 0.12, AMP * 1.12)
    ax.set_yticks([])
    ticks = nice_ticks(xrange[0], xrange[1])
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=T2)
    for s in ax.spines.values():
        s.set_visible(False)
    for xt in ticks:
        ax.axvline(xt, color=GRID, lw=0.8, alpha=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=10, color=T2, labelpad=8)
    ax.set_title(title, fontsize=12, pad=14, color=T1, fontweight="semibold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the frozen snapshot from the live TTSDS2 cache "
                         "(needs the ttsds venv + ~/.cache/ttsds + wav dirs)")
    args = ap.parse_args()

    if args.refresh:
        data = refresh_from_cache()
    elif FROZEN.exists():
        data = load_frozen()
    else:
        print(f"no snapshot at {FROZEN}; building it once from the TTSDS2 cache "
              f"(use --refresh to rebuild later)")
        data = refresh_from_cache()

    out_dir = REPO / "assets/plots"
    for bench_name, xlabel, xrange in BENCHES:
        rows, modes = data[bench_name]
        fig, axes = plt.subplots(1, len(SETS), figsize=(16, 9), facecolor=BG)
        if len(SETS) == 1:
            axes = [axes]
        for ax, (set_key, _, set_title) in zip(axes, SETS):
            ridgeline_ax(ax, rows, modes, set_key, xrange, xlabel, set_title)

        fig.suptitle("Zonos2 vs Reference TTS Models — Pitch Distribution Match",
                     fontsize=26 * 72 / 96, color=T1, fontweight="bold", x=0.5, y=0.975)
        fig.text(0.5, 0.925,
                 "Per-file mean-F0 distributions vs ground-truth reference (dashed); "
                 "rows sorted by Wasserstein-1 distance, lower is closer",
                 fontsize=13 * 72 / 96 + 4, color=T2, ha="center")
        fig.text(0.06, 0.025,
                 "TTSDS2 Pitch benchmark · ZTTS2-Eval EN test sets · ~100 utterances per system · "
                 "closed/API rows tagged ·clone (per-utt voice clone) or ·preset (fixed system voice) · "
                 "W1 computed on raw mean-F0 values",
                 fontsize=8.5, color=MUTED, ha="left")
        fig.tight_layout(rect=[0.04, 0.05, 0.97, 0.87])
        out = out_dir / "ttsds_pitch_ridgeline.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
