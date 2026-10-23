#!/usr/bin/env python3
"""Create a reproducible random 50/50 split of a reference wav directory."""
import argparse
import hashlib
import random
from pathlib import Path


def split_indices(n: int, seed: int, key: str) -> tuple[list[int], list[int]]:
    """Partition all indices into deterministic, randomly shuffled halves."""
    if n < 2:
        raise ValueError(f"need at least two items to split, got {n}")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    indices = list(range(n))
    rng.shuffle(indices)
    cut = n // 2
    return indices[:cut], indices[cut:]


def valid_wavs(src: Path) -> list[Path]:
    wavs = []
    for path in sorted(src.glob("*.wav")):
        try:
            if path.is_file() and path.stat().st_size > 44:
                wavs.append(path)
        except OSError:
            continue
    return wavs


def materialize_split(src: Path, left: Path, right: Path, seed: int, key: str) -> tuple[int, int]:
    wavs = valid_wavs(src)
    left_idx, right_idx = split_indices(len(wavs), seed, key)
    left.mkdir(parents=True, exist_ok=False)
    right.mkdir(parents=True, exist_ok=False)
    for out_dir, indices in ((left, left_idx), (right, right_idx)):
        for index in indices:
            wav = wavs[index]
            (out_dir / wav.name).symlink_to(wav.resolve())
    return len(left_idx), len(right_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", required=True, help="stable dataset key, e.g. zero_shot/en")
    args = ap.parse_args()
    n_left, n_right = materialize_split(
        Path(args.src), Path(args.left), Path(args.right), args.seed, args.key)
    print(f"split {args.src}: {n_left} + {n_right} wavs (seed={args.seed}, key={args.key})")


if __name__ == "__main__":
    main()
