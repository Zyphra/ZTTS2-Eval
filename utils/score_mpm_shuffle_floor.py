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
      --ref-dir   data/clean/en/waveform \\
      --synth-dir results/omni_qwen3-tts-1.7b/clean/en/wavs \\
      --synth-dir results/omni_voxcpm2/clean/en/wavs \\
      --out results/_mpm_shuffle_floor/clean_en.tsv

With no --synth-dir it just builds+caches the anchor and prints the split-GT
self-distance (d_ref floor) for reference.
"""
import argparse
import hashlib
import logging
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
import librosa  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.benchmark import BENCHMARK_VERSION  # noqa: E402

CACHE_DIR = Path(os.path.expanduser("~/.cache/ttsds/mpm_shuffle_floor"))
SR = 22050
# MPMBenchmark calls every measure with durations=[1000] (each keeps its first 1000
# frames); PitchMeasure / VoiceActivityMeasure run DIO at hop 256 (speed 1) at 22.05 kHz.
DURATIONS = np.array([1000])
FRAME_PERIOD_MS = 256 / SR * 1000
_FEAT = None  # per-worker feature extractors (see _feature_worker_init)


def _pitch_and_vad(f0):
    """TTSDS PitchMeasure and VoiceActivityMeasure from ONE stonemask contour.

    Both measures run the identical dio + stonemask on the same audio and differ only in
    post-processing, so the contour is computed once. The post-processing is theirs line
    for line (silence_mask=None), followed by the Measure.interpolate their __call__ applies."""
    from ttsds.util.measures import Measure
    pitch = f0.astype(np.float32)[: sum(DURATIONS)]
    pitch[pitch < 1e-6] = np.nan
    vad = pitch.copy()
    if np.isnan(pitch).all():
        pitch[:] = 1e-6
    vad[~np.isnan(vad)] = 1
    vad[np.isnan(vad)] = 0
    return Measure.interpolate(pitch), Measure.interpolate(vad)


def _tokens(bench, pitch, energy, vad):
    """Replicate mpm._get_distribution tokenization of the three measures -> (3, T) long tensor."""
    pitch, energy, vad = torch.tensor(pitch), torch.tensor(energy), torch.tensor(vad)
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


# --- Tokenization runs in a process pool ---------------------------------------------
# pyworld's dio + stonemask dominate the run time and are single-threaded, so the
# per-utterance tokens are computed on every CPU of the job while the parent embeds
# finished utterances on the GPU. The contour is computed once per utterance and shared
# by the pitch and VAD channels (_pitch_and_vad).
def _feature_worker_init(consts):
    global _FEAT
    from ttsds.util.measures import EnergyMeasure
    torch.set_num_threads(1)
    c = dict(consts)
    _FEAT = SimpleNamespace(energy_measure=EnergyMeasure(),
                            bins=torch.from_numpy(c.pop("bins")), **c)


# Leading/trailing silence is trimmed before featurization, on the reference and the
# systems alike, with the speaking-rate metric's own threshold. MPM's VAD and energy
# channels otherwise count those frames, and the FLEURS-R recordings carry ~2.45 s of
# them per utterance while most systems emit tightly trimmed audio: untrimmed, span
# tracked |system padding - real padding| at Spearman +0.80 within language (8-language
# probe) and ranked tightly trimmed systems near the scrambled floor. docs/controls_2026-08.md C6.
TRIM_TOP_DB = 35.0


def _load_wav(path):
    """ttsds DirectoryDataset.__getitem__ (22.05 kHz, pad to 16000 samples, peak-normalize)
    without its on-disk audio cache, after trimming leading/trailing silence (TRIM_TOP_DB).
    The ttsds cache keys on str.__hash__, which Python randomizes per process, so it
    never hits: it only wrote every decoded wav to ~/.cache/ttsds again on every run."""
    audio, _ = librosa.load(path, sr=SR)
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    if trimmed.size:
        audio = trimmed
    if audio.shape[0] < 16000:
        audio = np.pad(audio, (0, 16000 - audio.shape[0]))
    return audio / (np.max(np.abs(audio)) + 1e-6)


def _energy(wav):
    return _FEAT.energy_measure(wav, DURATIONS)["measure"]


def _f0_pyworld(wav):
    import pyworld as pw
    x = wav.astype(np.float64)
    f0, t = pw.dio(x, SR, frame_period=FRAME_PERIOD_MS, speed=1)
    return pw.stonemask(x, f0, t, SR)


def _tokens_for_path(path):
    wav = _load_wav(path)
    pitch, vad = _pitch_and_vad(_f0_pyworld(wav))
    return _tokens(_FEAT, pitch, _energy(wav), vad).numpy()


def _wav_files(wav_dir):
    # DirectoryDataset's own file order, so token lists (and the seeded shuffle and
    # split of the anchor) match what the TTSDS loader produced.
    return list(Path(wav_dir).rglob("*.wav"))


def _feature_pool(bench):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    consts = dict(pitch_min=bench.pitch_min, pitch_max=bench.pitch_max,
                  vad_min=bench.vad_min, vad_max=bench.vad_max, bins=bench.bins.numpy())
    # spawn: the parent holds a CUDA context, which forked children must not inherit.
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_feature_worker_init, initargs=(consts,))


def _collect_tokens(pool, wav_dir):
    return [torch.from_numpy(t)
            for t in pool.map(_tokens_for_path, _wav_files(wav_dir), chunksize=4)]


def _embed_dir(bench, pool, wav_dir):
    """Per-utterance embeddings of a dir, embedding each utterance as soon as its tokens
    arrive so the GPU works while the pool tokenizes the rest."""
    return [_embed(bench, torch.from_numpy(t))
            for t in pool.map(_tokens_for_path, _wav_files(wav_dir), chunksize=4)]


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


def mpm_span(d_ref, d_ref_floor, d_noise_floor):
    """Position of a system between real-vs-real (0) and contour-scrambled (1):
    (d_ref - d_ref_floor) / (d_noise_floor - d_ref_floor). >1 is WORSE than scrambled."""
    return (d_ref - d_ref_floor) / (d_noise_floor - d_ref_floor)


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
    # v5 = leading/trailing silence trimmed before featurization (TRIM_TOP_DB).
    h = hashlib.md5((content + f"|seed={seed}|v5").encode()).hexdigest()[:16]
    return h, len(wavs), sum(p.stat().st_size for p in wavs)


def _build_anchor(bench, pool, ref_dir, seed):
    """Return ((ref_mu, ref_sig), (noise_mu, noise_sig), d_ref_floor), caching
    the FID stats to disk (~0.5 MB) rather than the raw frame matrices."""
    from ttsds.util.distances import frechet_distance
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key, n_wavs, total_bytes = _key(ref_dir, seed)
    npz = CACHE_DIR / f"{key}_stats.npz"
    if npz.exists():
        print(f"[cache] anchor {key} (ref={ref_dir}, {n_wavs} wavs, "
              f"{total_bytes} bytes, seed={seed}; key covers names+sizes+mtimes)")
        z = np.load(npz)
        return (z["ref_mu"], z["ref_sig"]), (z["noise_mu"], z["noise_sig"]), float(z["d_ref_floor"])

    print(f"[build] anchor {key} from {ref_dir}")
    toks = _collect_tokens(pool, ref_dir)
    rng = np.random.default_rng(seed)
    # Embeddings are per utterance, so the split halves below reuse these rows instead
    # of embedding every reference utterance a second time.
    ref_embs = [_embed(bench, t) for t in toks]
    ref_stats = _stats(np.vstack(ref_embs))
    noise_stats = _stats(_stack_embed(bench, _shuffle_joint(toks, rng)))

    # Random split-GT self-distance for the finite-sample d_ref floor.
    split_rng = np.random.default_rng(seed)
    order = split_rng.permutation(len(toks))
    half = len(order) // 2
    a = np.vstack([ref_embs[i] for i in order[:half]])
    b = np.vstack([ref_embs[i] for i in order[half:]])
    d_ref_floor = frechet_distance(_stats(a), _stats(b))

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

    from ttsds.benchmarks.prosody.mpm import MPMBenchmark
    from ttsds.util.distances import frechet_distance
    bench = MPMBenchmark()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(bench, "_to_device"):
        bench._to_device(device)
    else:
        # Vanilla TTSDS2 does not expose GPU support on MPM, but this scorer
        # controls the input device in _embed(), so moving the model is sufficient.
        bench.device = device
        bench.model = bench.model.to(device)
    pool = _feature_pool(bench)
    ref_stats, noise_stats, d_ref_floor = _build_anchor(bench, pool, args.ref_dir, args.seed)
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
        print(f"[system] {label}: {sd}", flush=True)
        t0 = time.time()
        embs = _embed_dir(bench, pool, sd)
        if not embs:
            print(f"[skip] {label}: no usable audio")
            continue
        # One covariance pass over the ~1M x 256 frame matrix serves both distances and
        # the projection (frechet_distance computes the same mean/cov from a raw matrix).
        sys_stats = _stats(np.vstack(embs))
        print(f"  {len(embs)} utterances in {time.time() - t0:.0f}s", flush=True)
        d_ref = frechet_distance(sys_stats, ref_stats)
        d_noise = frechet_distance(sys_stats, noise_stats)
        score = d_noise / (d_ref + d_noise) * 100
        # position in [d_ref_floor .. d_noise_floor]: 0=indistinguishable, 1=fully scrambled
        span = mpm_span(d_ref, d_ref_floor, d_noise_floor)
        rows.append((label, d_ref, d_noise, score, span))
        sm, sv = _project(*sys_stats, U)
        pos, sd_n = _norm(sm, sv)
        dist_systems.append({"name": label, "mean": round(pos, 5), "std": round(sd_n, 5),
                             "span": round(span, 5),
                             "overlap": round(_ovl_1d(gt_pos, gt_sd, pos, sd_n), 5)})
    pool.shutdown()

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
            f.write("system\td_ref\td_noise\tscore\tspan\td_ref_floor\td_noise_floor\t"
                    "benchmark_version\n")
            for label, d_ref, d_noise, score, span in sorted(rows, key=lambda r: r[1]):
                f.write(f"{label}\t{d_ref:.6f}\t{d_noise:.6f}\t{score:.4f}\t{span:.4f}"
                        f"\t{d_ref_floor:.6f}\t{d_noise_floor:.6f}\t{BENCHMARK_VERSION}\n")
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
