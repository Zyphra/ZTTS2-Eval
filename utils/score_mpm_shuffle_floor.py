"""Score MPM (TTSDS Masked Prosody Model) against a *contour-scrambled* noise
floor instead of TTSDS's waveform-level noise.

Motivation: TTSDS's built-in MPM noise (all-ones/zeros/gaussian/uniform audio)
maps to a degenerate corner of MPM space ~840x farther from real speech than a
second real sample (d_ref~=0.28 vs d_noise~=233), so score=d_noise/(d_ref+d_noise)
saturates at ~99.9 and has no dynamic range. This scorer replaces that anchor
with `shuffle_joint`: a per-utterance frame permutation of the *reference* MPM
input tokens (pitch/energy/VAD bins), applied jointly to all three channels.
That destroys ONLY the temporal contour MPM is meant to encode while holding
every marginal and joint frame statistic fixed -- an on-manifold, contrastive
floor (~19 on en, ~69x d_ref) rather than an out-of-distribution artifact.

The reference distribution and the shuffle noise distribution are derived from
--ref-dir once and cached (keyed by the ref wavs' names+sizes+mtimes + seed, so
a rebuilt reference dir invalidates the anchor) and every system is scored
against the SAME fixed anchor.

Reported per system:
  d_ref   = frechet(system, reference)          -- raw, the discriminating axis
  d_noise = frechet(system, shuffle_joint(ref)) -- fixed floor
  score   = d_noise/(d_ref+d_noise)*100         -- bounded, but saturating
Real systems should land on the d_ref axis between ~d_ref_floor (indistinguish-
able from GT) and the noise distance (~fully contour-scrambled).

Usage:
  python utils/score_mpm_shuffle_floor.py \\
      --ref-dir   data/zero_shot/en/waveform \\
      --synth-dir results/omni_qwen3-tts-1.7b/zero_shot/en/wavs \\
      --synth-dir results/omni_voxcpm2/zero_shot/en/wavs \\
      --out results/_mpm_shuffle_floor/zero_shot_en.tsv

With no --synth-dir it just builds+caches the anchor and prints the split-GT
self-distance (d_ref floor) for reference.
"""
import argparse
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
import librosa  # noqa: E402
from ttsds.benchmarks.prosody.mpm import MPMBenchmark  # noqa: E402
from ttsds.util.dataset import DirectoryDataset  # noqa: E402
from ttsds.util.distances import frechet_distance  # noqa: E402

CACHE_DIR = Path(os.path.expanduser("~/.cache/ttsds/mpm_shuffle_floor"))


def _tokens_for_wav(bench, wav, sr):
    """Replicate mpm._get_distribution tokenization -> (3, T) long tensor."""
    if sr != 22050:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=22050)
    pitch = torch.tensor(bench.pitch_measure(wav, np.array([1000]))["measure"])
    energy = torch.tensor(bench.energy_measure(wav, np.array([1000]))["measure"])
    vad = torch.tensor(bench.voice_activity_measure(wav, np.array([1000]))["measure"])
    for t in (pitch, energy, vad):
        t[torch.isnan(t)] = -1000
    pitch = torch.clip(pitch, bench.pitch_min, bench.pitch_max) / (bench.pitch_max - bench.pitch_min)
    # v4: per-utterance p95 energy normalization (docs/mpm_itw_limits.md, "loudness
    # confound"). The upstream absolute 0-0.2 window makes energy tokens scale with
    # playback gain (span vs output loudness r = -0.979 across systems); e / p95(e)
    # clipped to [0, 1] is gain-invariant. p95 rather than max so a single click
    # cannot rescale the whole utterance. NaNs were set to -1000 above and clip to 0,
    # exactly as under the old absolute window.
    pos = energy[energy > 0]
    p95 = float(torch.quantile(pos, 0.95)) if pos.numel() else 1.0
    energy = torch.clip(energy / max(p95, 1e-8), 0.0, 1.0)
    vad = torch.clip(vad, bench.vad_min, bench.vad_max) / (bench.vad_max - bench.vad_min)
    pitch = torch.bucketize(pitch, bench.bins)
    energy = torch.bucketize(energy, bench.bins)
    vad = torch.bucketize(vad, torch.linspace(0, 1, 2))
    m = min(len(pitch), len(energy), len(vad))
    return torch.stack([pitch[:m], energy[:m], vad[:m]])


def _embed(bench, toks):
    x = toks.unsqueeze(0).to(bench.device)
    with torch.no_grad():
        r = bench.model(x, return_layer=bench.mpm_layer)["representations"]
    return r.detach().cpu().numpy()[0]


def _collect_tokens(bench, wav_dir):
    ds = DirectoryDataset(str(wav_dir), name=Path(wav_dir).name, sample_rate=22050)
    return [_tokens_for_wav(bench, wav, ds.sample_rate) for wav, _ in ds.iter_with_progress(bench)]


def _stack_embed(bench, toks_list):
    return np.vstack([_embed(bench, t) for t in toks_list])


def _stats(emb):
    """FID sufficient statistics: (mean, covariance). All frechet_distance
    needs -- storing these instead of the raw (~1M x 256) frame matrix drops
    the cached anchor from ~1 GB to ~0.5 MB with bit-identical distances."""
    return np.mean(emb, axis=0), np.cov(emb, rowvar=False)


def _project(mu, sig, U):
    """Project a Gaussian (mu, Σ) onto unit axis U -> (mean_1d, var_1d)."""
    return float(mu @ U), float(U @ sig @ U)


def _ovl_1d(m1, s1, m2, s2, n=4000):
    """Overlap coefficient (shared area) of two 1-D Gaussians, numerically."""
    s1, s2 = max(s1, 1e-9), max(s2, 1e-9)
    lo = min(m1 - 5 * s1, m2 - 5 * s2)
    hi = max(m1 + 5 * s1, m2 + 5 * s2)
    x = np.linspace(lo, hi, n)
    d1 = np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
    d2 = np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
    mn = np.minimum(d1, d2)
    dx = (hi - lo) / (n - 1)                       # trapezoid (np.trapz gone in numpy 2)
    return float((mn.sum() - 0.5 * (mn[0] + mn[-1])) * dx)


def _shuffle_joint(toks_list, rng):
    out = []
    for t in toks_list:
        p = torch.from_numpy(rng.permutation(t.shape[1]))
        out.append(t[:, p])
    return out


def _key(ref_dir, seed):
    """Anchor cache key over the reference dir CONTENTS (names + sizes + mtimes),
    not just the file names: a reference dir rebuilt under the same names (dataset
    repair, resampling fix, enriched-pkl reconstruction) must not silently reuse a
    stale anchor. Returns (key, n_wavs, total_bytes) so a cache hit can print the
    inputs it was keyed on."""
    wavs = sorted(p for p in Path(ref_dir).iterdir() if p.suffix.lower() == ".wav")
    content = "\n".join(f"{p.name}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}"
                        for p in wavs)
    # v3 = random (rather than sorted/interleaved) split-GT floor.
    # v4 = per-utterance p95 energy normalization (the featurization is part of the
    #      key by version tag, since the content hash cannot see code changes).
    h = hashlib.md5((content + f"|seed={seed}|v4").encode()).hexdigest()[:16]
    return h, len(wavs), sum(p.stat().st_size for p in wavs)


def _build_anchor(bench, ref_dir, seed):
    """Return ((ref_mu, ref_sig), (noise_mu, noise_sig), d_ref_floor), caching
    the FID stats to disk (~0.5 MB) rather than the raw frame matrices."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key, n_wavs, total_bytes = _key(ref_dir, seed)
    npz = CACHE_DIR / f"{key}_stats.npz"
    if npz.exists():
        print(f"[cache] anchor {key} (ref={ref_dir}, {n_wavs} wavs, "
              f"{total_bytes} bytes, seed={seed}; key covers names+sizes+mtimes)")
        z = np.load(npz)
        return (z["ref_mu"], z["ref_sig"]), (z["noise_mu"], z["noise_sig"]), float(z["d_ref_floor"])

    print(f"[build] anchor {key} from {ref_dir}")
    toks = _collect_tokens(bench, ref_dir)
    rng = np.random.default_rng(seed)
    ref_stats = _stats(_stack_embed(bench, toks))
    noise_stats = _stats(_stack_embed(bench, _shuffle_joint(toks, rng)))

    # Random split-GT self-distance for the finite-sample d_ref floor.
    split_rng = np.random.default_rng(seed)
    order = split_rng.permutation(len(toks))
    half = len(order) // 2
    a = [toks[i] for i in order[:half]]
    b = [toks[i] for i in order[half:]]
    d_ref_floor = frechet_distance(_stats(_stack_embed(bench, a)),
                                   _stats(_stack_embed(bench, b)))

    np.savez(npz, ref_mu=ref_stats[0], ref_sig=ref_stats[1],
             noise_mu=noise_stats[0], noise_sig=noise_stats[1],
             d_ref_floor=np.float64(d_ref_floor))
    return ref_stats, noise_stats, d_ref_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--synth-dir", action="append", default=[],
                    help="system wav dir; repeatable. Omit to just build the anchor.")
    ap.add_argument("--label", action="append", default=[],
                    help="label per --synth-dir (defaults to results/<model>/... 2nd path segment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional TSV output path")
    ap.add_argument("--dist-out", default=None,
                    help="optional JSON: GT / scramble / per-system 1-D distributions "
                    "projected onto the GT->scramble axis, for the overlap panel")
    ap.add_argument("--lang", default=None, help="language tag for --dist-out")
    args = ap.parse_args()

    bench = MPMBenchmark()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(bench, "_to_device"):
        bench._to_device(device)
    else:
        # Vanilla TTSDS2 does not expose GPU support on MPM, but this scorer
        # controls the input device in _embed(), so moving the model is sufficient.
        bench.device = device
        bench.model = bench.model.to(device)
    ref_stats, noise_stats, d_ref_floor = _build_anchor(bench, args.ref_dir, args.seed)
    d_noise_floor = frechet_distance(ref_stats, noise_stats)  # ref-vs-noise, for context
    print(f"\nanchor: ref={args.ref_dir}")
    print(f"  d_ref_floor (split-GT real-vs-real) = {d_ref_floor:.6f}")
    print(f"  d_noise      (ref vs shuffle_joint) = {d_noise_floor:.6f}")

    # 1-D projection onto the GT->scramble axis (the direction prosody-realism lives
    # on), normalized so GT mean=0 and scramble mean=1. Lets the explorer draw each
    # system's prosody distribution as a bell against GT's and read the overlap.
    U = noise_stats[0] - ref_stats[0]
    L = float(np.linalg.norm(U)) or 1.0
    U = U / L
    gt_m, gt_v = _project(ref_stats[0], ref_stats[1], U)
    ns_m, ns_v = _project(noise_stats[0], noise_stats[1], U)

    def _norm(m1d, v1d):
        return (m1d - gt_m) / L, (max(v1d, 0.0) ** 0.5) / L

    gt_pos, gt_sd = _norm(gt_m, gt_v)          # (0.0, GT spread)
    ns_pos, ns_sd = _norm(ns_m, ns_v)          # (~1.0, scramble spread)

    rows, dist_systems = [], []
    for i, sd in enumerate(args.synth_dir):
        if args.label and i < len(args.label):
            label = args.label[i]
        else:
            parts = Path(sd).parts
            label = parts[parts.index("results") + 1] if "results" in parts else Path(sd).name
        print(f"[system] {label}: {sd}")
        tokens = _collect_tokens(bench, sd)
        if not tokens:
            print(f"[skip] {label}: no usable audio")
            continue
        sys_emb = _stack_embed(bench, tokens)
        d_ref = frechet_distance(sys_emb, ref_stats)
        d_noise = frechet_distance(sys_emb, noise_stats)
        score = d_noise / (d_ref + d_noise) * 100
        # position in [d_ref_floor .. d_noise_floor]: 0=indistinguishable, 1=fully scrambled
        span = (d_ref - d_ref_floor) / (d_noise_floor - d_ref_floor)
        rows.append((label, d_ref, d_noise, score, span))
        sm, sv = _project(*_stats(sys_emb), U)
        pos, sd_n = _norm(sm, sv)
        dist_systems.append({"name": label, "mean": round(pos, 5), "std": round(sd_n, 5),
                             "span": round(span, 5),
                             "overlap": round(_ovl_1d(gt_pos, gt_sd, pos, sd_n), 5)})

    if rows:
        print(f"\n{'system':<24}{'d_ref':>10}{'d_noise':>12}{'score':>9}{'span':>9}")
        print("-" * 64)
        for label, d_ref, d_noise, score, span in sorted(rows, key=lambda r: r[1]):
            print(f"{label:<24}{d_ref:>10.4f}{d_noise:>12.4f}{score:>9.2f}{span:>9.3f}")
        print(f"\nspan = (d_ref - {d_ref_floor:.3f}) / ({d_noise_floor:.3f} - {d_ref_floor:.3f})"
              f"  [0=GT-like, 1=contour-scrambled]")

    if args.out and rows:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            f.write("system\td_ref\td_noise\tscore\tspan\td_ref_floor\td_noise_floor\n")
            for label, d_ref, d_noise, score, span in sorted(rows, key=lambda r: r[1]):
                f.write(f"{label}\t{d_ref:.6f}\t{d_noise:.6f}\t{score:.4f}\t{span:.4f}"
                        f"\t{d_ref_floor:.6f}\t{d_noise_floor:.6f}\n")
        print(f"\nwrote {outp}")

    if args.dist_out and dist_systems:
        import json
        lang = args.lang or (Path(args.out).stem if args.out else None)
        dist = {
            "lang": lang,
            "axis": "GT->scramble (0 = GT mean, 1 = scramble mean, along the FID axis)",
            "gt": {"mean": round(gt_pos, 5), "std": round(gt_sd, 5)},
            "scramble": {"mean": round(ns_pos, 5), "std": round(ns_sd, 5)},
            "d_ref_floor": round(float(d_ref_floor), 6),
            "systems": sorted(dist_systems, key=lambda r: r["mean"]),
        }
        outp = Path(args.dist_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(dist, f, indent=1)
        print(f"wrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
