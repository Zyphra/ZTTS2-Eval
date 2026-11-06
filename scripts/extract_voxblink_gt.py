"""Extract the held-out target recordings for the in-the-wild eval.

Stage-B produced `data/in-the-wild/<lang>/{text, prompt_text, prompt_wav.scp,
waveform/prompt_audio_*.wav}` from a stratified pick of `_stage_a/manifest.tsv`. The
TARGET audio (the real recording behind `text`, same speaker as the prompt, different
video) is the reference for held-out speaker similarity and for the ground-truth
baseline. It ships on the HF dataset Zyphra/ZTTS2-Eval; this script rebuilds it from the
raw VoxBlink2 audio.

For each lang dir, this script:
  1. reads `prompt_text` (uid -> prompt transcript) and the per-lang slice of
     `_stage_a/manifest.tsv`,
  2. joins on prompt_text to recover each uid's target_{vid,seg,start,end},
  3. extracts the target audio slice from the local VoxBlink2 audio tree
     $VOXBLINK2_AUDIO_ROOT/<spk>/<vid>/<seg>.wav
  4. writes <data_root>/<lang>/target_wav/<uid>.wav (16 kHz PCM16, mono).

Then validate and write target_wav.scp / target_provenance.json with:
    python scripts/build_voxblink_targets.py

Usage:
    python scripts/extract_voxblink_gt.py [--data-root data/in-the-wild]
"""
import argparse
import csv
import os
import sys

import soundfile as sf

# Local VoxBlink2 audio tree (<spk>/<vid>/<seg>.wav). Set VOXBLINK2_AUDIO_ROOT to
# point at your copy of the dataset.
AUDIO_ROOT = os.environ.get("VOXBLINK2_AUDIO_ROOT", "/path/to/VoxBlink2/audio")
DATA_ROOT = "data/in-the-wild"
SR = 16000  # stage_b writes 16 kHz PCM16; keep consistent.


def wav_path(spk, vid, seg):
    return f"{AUDIO_ROOT}/{spk}/{vid}/{seg}.wav"


def load_text_file(path):
    """Parse '<uid> <value>' file -> ordered list of (uid, value) and dict."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, val = line.split(" ", 1)
            items.append((uid, val))
    return items


def read_slice(path, start, end):
    info = sf.info(path)
    s = int(start * info.samplerate)
    e = int(end * info.samplerate)
    wav, sr = sf.read(path, start=s, stop=e, dtype="float32", always_2d=False)
    if sr != SR:
        # All voxblink source audio is 16 kHz mono per stage_a; warn if not.
        print(f"  WARN: {path} is {sr} Hz (expected {SR})", file=sys.stderr)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--manifest", default=os.path.join(DATA_ROOT, "_stage_a/manifest.tsv"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Group manifest rows by (lang, prompt_text). prompt_text is the join key.
    # Stage-B selects N rows per lang from the manifest; their prompt_text values
    # are unique within a lang (each comes from a different speaker/segment).
    rows_by_key = {}
    with open(args.manifest, newline="", encoding="utf-8") as f:
        # QUOTE_NONE: manifest is raw tab-separated; a '"' in a transcript must not
        # open a quoted field (which would swallow rows and misalign columns).
        rdr = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in rdr:
            rows_by_key.setdefault((row["lang"], row["prompt_text"]), []).append(row)

    overall_ok = overall_fail = overall_dup = 0
    for lang in sorted(os.listdir(args.data_root)):
        lang_dir = os.path.join(args.data_root, lang)
        prompt_text_path = os.path.join(lang_dir, "prompt_text")
        if not os.path.isfile(prompt_text_path):
            continue

        out_dir = os.path.join(args.data_root, lang, "target_wav")
        os.makedirs(out_dir, exist_ok=True)

        items = load_text_file(prompt_text_path)
        ok = fail = skip = dup = 0
        for uid, ptxt in items:
            out_wav = os.path.join(out_dir, f"{uid}.wav")
            if os.path.exists(out_wav) and not args.overwrite:
                skip += 1
                continue
            matches = rows_by_key.get((lang, ptxt))
            if not matches:
                print(f"  [miss] {lang}/{uid}: no manifest row for prompt_text",
                      file=sys.stderr)
                fail += 1
                continue
            # Short stock phrases DO recur across speakers in the manifest, so a
            # (lang, prompt_text) collision can point at a different speaker's
            # target audio. Deterministic policy: same-speaker duplicates are safe
            # (Stage A picks one target per (lang, speaker), so all rows agree on
            # the target) -> take the first, with a warning; cross-speaker
            # duplicates are ambiguous -> skip rather than risk wrong-speaker GT.
            if len(matches) > 1:
                speakers = sorted({m["speaker"] for m in matches})
                if len(speakers) > 1:
                    print(f"  [dup] {lang}/{uid}: prompt_text matches "
                          f"{len(matches)} manifest rows across {len(speakers)} "
                          f"speakers {speakers}; skipping (ambiguous GT)",
                          file=sys.stderr)
                    dup += 1
                    continue
                print(f"  [dup] {lang}/{uid}: prompt_text matches {len(matches)} "
                      f"rows for speaker {speakers[0]}; using the first",
                      file=sys.stderr)
            row = matches[0]
            try:
                wav, sr = read_slice(
                    wav_path(row["speaker"], row["target_vid"], row["target_seg"]),
                    float(row["target_start"]), float(row["target_end"]))
                if wav.size < SR // 2:  # < 0.5 s -> probably broken
                    print(f"  [short] {lang}/{uid}: only {wav.size} samples",
                          file=sys.stderr)
                    fail += 1
                    continue
                sf.write(out_wav, wav, sr, subtype="PCM_16")
                ok += 1
            except Exception as e:
                print(f"  [err] {lang}/{uid}: {e}", file=sys.stderr)
                fail += 1
        print(f"  {lang}: ok={ok} skip={skip} fail={fail} dup_skipped={dup}")
        overall_ok += ok; overall_fail += fail; overall_dup += dup

    print(f"\ndone: {overall_ok} ok, {overall_fail} failed, {overall_dup} "
          f"ambiguous-dup skipped across all langs -> {args.data_root}/<lang>/target_wav/")


if __name__ == "__main__":
    main()
