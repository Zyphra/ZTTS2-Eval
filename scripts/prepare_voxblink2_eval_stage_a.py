"""Stage A of the VoxBlink2-derived eval: pick candidate (prompt, target) pairs
from the enriched VoxBlink2 candidate pool and emit a manifest + a list of
directories that need `o+rX` before Stage B can read the audio.

Inputs
  $VOXBLINK2_ENRICHED (default ./voxblink2_enriched.pkl) — (rows, lang_counts)
    where each row is (spk, vid, seg, start, end, dur, xtx, lang, lang_prob, text)

Outputs (relative to --output-dir)
  manifest.tsv          — one (lang, speaker, prompt-segment, target-segment) per row
  dirs_to_chmod.txt     — deduped (speaker_dir, prompt_video_dir, target_video_dir)
  files_to_chmod.txt    — the prompt/target wav files (group/other-restricted too)

Workflow
  1. python scripts/prepare_voxblink2_eval_stage_a.py
  2. xargs -d'\\n' -n1 -P16 chmod o+rX < .../dirs_to_chmod.txt   # run as audio owner
     xargs -d'\\n' -n1 -P16 chmod o+r  < .../files_to_chmod.txt
  3. python scripts/prepare_voxblink2_eval_stage_b.py    (computes MSR-UTMOS, decodes,
                                                          stratified-samples up to
                                                          ~1500 prompts/lang)
"""
import argparse
import collections
import os
import pickle
import random
import sys

# Local VoxBlink2 audio tree (<spk>/<vid>/<seg>.wav). Set VOXBLINK2_AUDIO_ROOT to
# point at your copy of the dataset.
AUDIO_ROOT = os.environ.get("VOXBLINK2_AUDIO_ROOT", "/path/to/VoxBlink2/audio")

# Languages from the existing clean subset (parity), plus the most populous
# additional VoxBlink2 languages. The last 6 are v2 additions with enough
# high-agreement speakers in the enriched pool to sustain a large sample.
DEFAULT_LANGS = [
    "en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru",
    "pt", "ar", "hi", "tr", "tl", "pl", "id", "th",
    "nl", "uk", "vi",
]


def load_enriched(path):
    with open(path, "rb") as f:
        rows, _ = pickle.load(f)
    return rows


def pick_speakers(rows, langs, min_xtx, min_lang_prob, min_dur, max_dur,
                  speakers_per_lang, candidates_per_speaker, seed):
    """Per (lang, speaker), pick one target video + N prompt-candidate videos.
    Returns list of manifest dicts."""
    by_lang_spk_vid = collections.defaultdict(  # lang -> spk -> vid -> [row]
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    lang_set = set(langs)
    for r in rows:
        spk, vid, seg, s, e, d, x, lang, prob, text = r
        if lang not in lang_set: continue
        if x < min_xtx: continue
        if prob < min_lang_prob: continue
        if not (min_dur <= d <= max_dur): continue
        if not text or len(text.strip()) < 3: continue
        by_lang_spk_vid[lang][spk][vid].append(r)

    manifest = []
    for lang in langs:
        # Per-lang RNG (string seeds hash deterministically in random.Random, unlike
        # builtin hash()), so the sample is reproducible under --seed and one
        # language's draw count cannot shift another language's picks.
        rnd = random.Random(f"{seed}:{lang}")
        spk_pool = [(spk, vmap) for spk, vmap in by_lang_spk_vid[lang].items()
                    if len(vmap) >= 2]
        rnd.shuffle(spk_pool)
        taken = 0
        n_before = len(manifest)
        for spk, vmap in spk_pool:
            if taken >= speakers_per_lang: break
            vids = list(vmap.keys())
            rnd.shuffle(vids)
            # target = one video; prompt-candidates = up to N other videos
            target_vid = vids[0]
            prompt_vids = vids[1:1 + candidates_per_speaker]
            if not prompt_vids: continue
            # target seg: pick the longest qualifying segment in the target video
            target = max(vmap[target_vid], key=lambda r: r[5])
            for pvid in prompt_vids:
                # prompt seg: random among qualifying segments in this video
                prompt = rnd.choice(vmap[pvid])
                manifest.append({
                    "lang": lang,
                    "speaker": spk,
                    "prompt_vid": pvid,
                    "prompt_seg": prompt[2],
                    "prompt_start": prompt[3],
                    "prompt_end": prompt[4],
                    "prompt_text": prompt[9],
                    "target_vid": target_vid,
                    "target_seg": target[2],
                    "target_start": target[3],
                    "target_end": target[4],
                    "target_text": target[9],
                })
            taken += 1
        print(f"  {lang}: {taken} speakers, {len(manifest) - n_before} candidates",
              file=sys.stderr)
    return manifest


def write_outputs(manifest, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, "manifest.tsv")
    dirs_path = os.path.join(output_dir, "dirs_to_chmod.txt")

    cols = ["lang", "speaker", "prompt_vid", "prompt_seg", "prompt_start",
            "prompt_end", "prompt_text", "target_vid", "target_seg",
            "target_start", "target_end", "target_text"]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for m in manifest:
            text_p = " ".join(m["prompt_text"].split())
            text_t = " ".join(m["target_text"].split())
            f.write("\t".join([
                m["lang"], m["speaker"], m["prompt_vid"], m["prompt_seg"],
                f"{m['prompt_start']:.3f}", f"{m['prompt_end']:.3f}", text_p,
                m["target_vid"], m["target_seg"],
                f"{m['target_start']:.3f}", f"{m['target_end']:.3f}", text_t,
            ]) + "\n")

    dirs = set()
    files = set()
    for m in manifest:
        spk_dir = f"{AUDIO_ROOT}/{m['speaker']}"
        dirs.add(spk_dir)
        dirs.add(f"{spk_dir}/{m['prompt_vid']}")
        dirs.add(f"{spk_dir}/{m['target_vid']}")
        # The prompt/target wav files themselves are also group/other-restricted;
        # Stage B and extract_voxblink_gt need them readable.
        files.add(f"{spk_dir}/{m['prompt_vid']}/{m['prompt_seg']}.wav")
        files.add(f"{spk_dir}/{m['target_vid']}/{m['target_seg']}.wav")
    with open(dirs_path, "w") as f:
        for d in sorted(dirs):
            f.write(d + "\n")
    files_path = os.path.join(output_dir, "files_to_chmod.txt")
    with open(files_path, "w") as f:
        for p in sorted(files):
            f.write(p + "\n")

    print(f"\nwrote {tsv_path}  ({len(manifest):,} rows)")
    print(f"wrote {dirs_path}  ({len(dirs):,} dirs)")
    print(f"wrote {files_path}  ({len(files):,} files)")
    print(f"\nnext (run as the audio-tree owner if you lack chmod rights):")
    print(f"  xargs -d'\\n' -n1 -P16 chmod o+rX < {dirs_path}")
    print(f"  xargs -d'\\n' -n1 -P16 chmod o+r  < {files_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", default=os.environ.get("VOXBLINK2_ENRICHED"),
                    required="VOXBLINK2_ENRICHED" not in os.environ)
    ap.add_argument("--output-dir", default="data/in-the-wild/_stage_a")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--speakers-per-lang", type=int, default=2000,
                    help="Cross-video speakers to draw per language (Stage B keeps "
                         "one prompt each, up to its --target-total).")
    ap.add_argument("--candidates-per-speaker", type=int, default=2,
                    help="Prompt-video candidates per speaker. Stage B picks 1 "
                         "after MSR-UTMOS stratification; >1 helps fill sparse bins.")
    ap.add_argument("--min-xtx", type=float, default=0.85)
    ap.add_argument("--min-lang-prob", type=float, default=0.85)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Order-preserving dedupe: iteration order feeds the RNG, so a set here made
    # the manifest depend on PYTHONHASHSEED despite --seed.
    langs = list(dict.fromkeys(args.langs.split(",")))
    print(f"loading {args.enriched}", file=sys.stderr)
    rows = load_enriched(args.enriched)
    print(f"loaded {len(rows):,} rows", file=sys.stderr)

    manifest = pick_speakers(
        rows, langs,
        min_xtx=args.min_xtx,
        min_lang_prob=args.min_lang_prob,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        speakers_per_lang=args.speakers_per_lang,
        candidates_per_speaker=args.candidates_per_speaker,
        seed=args.seed,
    )
    write_outputs(manifest, args.output_dir)


if __name__ == "__main__":
    main()
