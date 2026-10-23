"""Score TTSDS2 pitch-distribution similarity for one synthetic corpus.

The retained TTSDS metric is Pitch: each utterance contributes its mean F0 and
TTSDS compares the synthetic and reference distributions with 1-D Wasserstein
distance, normalized against its noise references. Higher scores are closer to
the real-speech reference.

Usage:
    python utils/score_ttsds_pitch.py \
        --synth-dir results/omni_qwen3-tts-1.7b/zero_shot/en/wavs \
        --ref-dir data/zero_shot/en/waveform \
        --label omni_qwen3-tts-1.7b \
        --out results/omni_qwen3-tts-1.7b/zero_shot/en/ttsds_pitch.csv
"""
import argparse
import hashlib
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from ttsds import BenchmarkSuite  # noqa: E402
from ttsds.benchmarks.prosody.pitch import PitchBenchmark  # noqa: E402
from ttsds.util.dataset import DirectoryDataset  # noqa: E402


BENCHMARKS = {"pitch": PitchBenchmark}


def _select_wavs(src_dir: Path, n: int) -> list:
    """The first N (all when n<=0) non-empty wavs of a directory, sorted by name."""
    candidates = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".wav")
    wavs = [p for p in candidates if p.is_file() and p.stat().st_size > 44]
    if not wavs:
        raise SystemExit(f"no non-empty wavs found in {src_dir}")
    return wavs if n <= 0 else wavs[:n]


def _content_dir(wavs: list, cache_dir: str, tag: str) -> Path:
    """Mirror `wavs` (symlinks) into a content-addressed dir under the cache.

    TTSDS caches each distribution keyed on DirectoryDataset.__hash__, which
    covers only the class, root_dir and sample params -- never file contents.
    Scoring straight from a results/.../wavs path therefore silently reuses a
    STALE cached distribution after the wavs are regenerated in place (the
    documented 'stale scores after --overwrite' pitfall), while --limit runs
    through a fresh mkdtemp path never hit the cache at all. Keying the dataset
    root PATH by content (sorted names + sizes + mtimes) fixes both: changed
    audio -> new path -> cache miss; identical audio -> same path -> cache hit.
    """
    fp = hashlib.md5("\n".join(
        f"{p.name}\t{p.stat().st_size}\t{p.stat().st_mtime_ns}" for p in wavs
    ).encode()).hexdigest()[:16]
    tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag)[:48]
    d = Path(cache_dir) / "content_dirs" / f"{tag}_{fp}"
    if d.is_dir():
        if sorted(p.name for p in d.iterdir()) == [p.name for p in wavs]:
            return d
        shutil.rmtree(d)  # hash collision on a changed selection -- rebuild
    d.mkdir(parents=True, exist_ok=True)
    for wav in wavs:
        (d / wav.name).symlink_to(wav.resolve())
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True, help="directory of generated wavs")
    ap.add_argument("--ref-dir", required=True, help="directory of reference wavs")
    ap.add_argument("--label", required=True, help="synthetic dataset label written to the CSV")
    ap.add_argument("--ref-label", default="prompt_audio_en",
                    help="reference dataset label written to the CSV")
    ap.add_argument("--out", required=True, help="output TTSDS pitch CSV")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/ttsds"),
                    help="TTSDS distribution cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap synthetic and reference inputs to their first N wavs")
    args = ap.parse_args()

    synth_dir = Path(args.synth_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    # Content-addressed symlink mirrors: the dataset identity (and so the TTSDS
    # distribution cache key) follows the audio contents, not the on-disk path.
    # The dirs persist under the cache -- they ARE the cache identity; a rebuilt
    # wav dir simply hashes to a new one and orphans the old few-KB dir.
    synth_eff = _content_dir(_select_wavs(synth_dir, args.limit), args.cache_dir, args.label)
    ref_eff = _content_dir(_select_wavs(ref_dir, args.limit), args.cache_dir, args.ref_label)
    print(f"[cache] synth dataset dir: {synth_eff}")
    print(f"[cache] ref   dataset dir: {ref_eff}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    synth_ds = DirectoryDataset(str(synth_eff), name=args.label, sample_rate=22050)
    ref_ds = DirectoryDataset(str(ref_eff), name=args.ref_label, sample_rate=22050)

    # BenchmarkSuite wires HuBERT clustering datasets even when only Pitch is
    # enabled. Dataset.sample() mutates in place, so supply throwaway datasets
    # to keep the full reference distribution intact.
    def cluster_dataset(suffix: str):
        dataset = DirectoryDataset(
            str(ref_eff),
            name=f"{args.ref_label}_cluster_{suffix}",
            sample_rate=22050,
        )
        dataset.sample(min(100, len(dataset)))
        return dataset

    suite = BenchmarkSuite(
        datasets=[synth_ds],
        reference_datasets=[ref_ds],
        benchmarks=BENCHMARKS,
        benchmark_kwargs={
            "hubert_token_sr": {"cluster_datasets": [cluster_dataset("sr")]},
            "hubert_token": {"cluster_datasets": [cluster_dataset("token")]},
        },
        device="cpu",
        cache_dir=args.cache_dir,
        include_environment=False,
        write_to_file=str(out_path),
    )
    rows = suite.run()
    print(f"wrote TTSDS Pitch CSV: {out_path} ({len(rows)} rows)")
    print(suite.get_aggregated_results().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
