"""Rebuild the enriched VoxBlink2 candidate pool (`voxblink2_enriched.pkl`) that
`prepare_voxblink2_eval_stage_a.py` consumes.

The pool is a join of two pre-computed artifacts over the VoxBlink2 audio tree:

  1. Cross-ASR-agreement (xtx) scores — a flat JSON dict keyed by
       voxblink2/<...>/audio/<spk>/<vid>/<seg>###<start>_<end>  ->  xtx  (0..1)
     (--xtx-json, ~34 GB; only the
     `voxblink2/` keys are used).
  2. Collapsed transcripts — one JSON per segment at
       <transcripts-root>/voxblink2/<...>/audio/<spk>/<vid>/<seg>.json
     whose `pipeline_v1` entries carry, per VAD span, multi-ASR transcripts plus
     `whisper_language` / `whisper_language_prob`.

Output pickle is `(rows, lang_counts)` where each row is
  (spk, vid, seg, start, end, dur, xtx, lang, lang_prob, text)
exactly as Stage A's `load_enriched` expects. `text` is the Voxtral transcript
(the field Stage A's shipped manifest was built from); `lang`/`lang_prob` are the
Whisper language id/probability; `xtx` is the agreement score.

To keep the join cheap we pre-filter on the two fields available from the xtx key
alone (xtx and duration) with looser bounds than Stage A's gates, so Stage A still
selects an identical set at its own thresholds. Only the surviving segments' small
transcript JSONs are read (multiprocess), instead of all ~2.3M.

Example:
  python scripts/build_voxblink2_enriched.py --out voxblink2_enriched.pkl
  # optionally reuse a pre-extracted grep of the xtx keys:
  LC_ALL=C grep -aoE '"voxblink2/[^"]+": ?-?[0-9.]+' segment_xtx_scores.json > vox_xtx.txt
  python scripts/build_voxblink2_enriched.py --xtx-cache vox_xtx.txt --out voxblink2_enriched.pkl
"""
import argparse
import collections
import json
import os
import pickle
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

VOX_PREFIX = "voxblink2/"
AUDIO_MARK = "/audio/"                       # spk/vid/seg follow the last /audio/
VOXTRAL = "mistralai/Voxtral-Mini-3B-2507"   # the ASR field used as `text`

# ---------------------------------------------------------------------------
# Transcript-language gate. Whisper LID labels the AUDIO; nothing ever checked
# that the TRANSCRIPT is in that language, and on accented speech LID fails
# systematically: the benchmark shipped a `yo` (Yoruba) set whose 148 transcripts
# were all fluent Nigerian English (removed 2026-08-29).
# Two cheap checks, no third-party LID dependency:
#   - non-Latin-script languages: the text must actually use that script;
#   - Latin-script languages other than English: reject texts dominated by
#     English function words (the observed failure class).
# ---------------------------------------------------------------------------
_SCRIPT_RANGES = {
    "zh": (0x4E00, 0x9FFF), "ja": (0x3040, 0x30FF), "ko": (0xAC00, 0xD7AF),
    "ru": (0x0400, 0x04FF), "uk": (0x0400, 0x04FF), "bg": (0x0400, 0x04FF),
    "ar": (0x0600, 0x06FF), "fa": (0x0600, 0x06FF), "he": (0x0590, 0x05FF),
    "hi": (0x0900, 0x097F), "th": (0x0E00, 0x0E7F), "el": (0x0370, 0x03FF),
}
# English function words chosen to NOT collide with German/Dutch/Romance
# homographs (no "was", "is", "in", "so", "of", "will", "on", "at").
_EN_STOPWORDS = frozenset(
    "the you that this they have for not with are but what your from just "
    "like know going don't i'm it's and we can to do my me be all about "
    "when how i a".split())


def transcript_lang_ok(lang, text):
    """True when the transcript is plausibly in `lang`; False = LID mislabel.

    Tuned to catch WHOLESALE mislabels (transcript in a different language), not
    code-switching: heavy Hinglish / brand-name-laden zh lines are genuine
    in-the-wild speech the set deliberately represents (cf. `latin_frac`)."""
    if lang in _SCRIPT_RANGES:
        lo, hi = _SCRIPT_RANGES[lang]
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return False
        in_script = sum(1 for c in alpha if lo <= ord(c) <= hi)
        # ja mixes kana with CJK ideographs; count both.
        if lang == "ja":
            in_script += sum(1 for c in alpha if 0x4E00 <= ord(c) <= 0x9FFF)
        return in_script / len(alpha) >= 0.3
    if lang != "en":
        words = [w.lower().strip(".,!?\"'") for w in text.split()]
        if len(words) >= 5:
            en = sum(1 for w in words if w in _EN_STOPWORDS)
            if en / len(words) >= 0.3:       # English function words dominate
                return False
    return True

# One xtx entry:  "voxblink2/.../audio/<spk>/<vid>/<seg>###<start>_<end>": <float>
XTX_LINE = re.compile(r'"(voxblink2/[^"]+)":\s*(-?[0-9.]+)')


def parse_key(key):
    """voxblink2/.../audio/<spk>/<vid>/<seg>###<start>_<end> ->
    (rel_no_ext, spk, vid, seg, start, end) or None if malformed.

    rel_no_ext is the transcript path relative to --transcripts-root (no .json)."""
    head, sep, se = key.partition("###")
    if not sep:
        return None
    i = head.rfind(AUDIO_MARK)
    if i < 0:
        return None
    parts = head[i + len(AUDIO_MARK):].split("/")
    if len(parts) != 3:
        return None
    spk, vid, seg = parts
    a, sep2, b = se.partition("_")
    if not sep2:
        return None
    try:
        start, end = float(a), float(b)
    except ValueError:
        return None
    return head, spk, vid, seg, start, end


def iter_xtx(xtx_json, xtx_cache):
    """Yield (key, xtx) for every voxblink2 entry.

    Uses a pre-extracted cache file if given, else streams `grep` over the big
    JSON (far faster than parsing 34 GB in Python; grep only emits voxblink2 keys).
    """
    if xtx_cache:
        with open(xtx_cache, encoding="utf-8") as f:
            for line in f:
                m = XTX_LINE.search(line)
                if m:
                    yield m.group(1), float(m.group(2))
        return
    cmd = ["grep", "-aoE", r'"voxblink2/[^"]+": ?-?[0-9.]+', xtx_json]
    env = dict(os.environ, LC_ALL="C")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, text=True,
                            bufsize=1 << 20)
    for line in proc.stdout:
        m = XTX_LINE.search(line)
        if m:
            yield m.group(1), float(m.group(2))
    proc.wait()


def read_transcript(args):
    """Worker: read one segment transcript, emit rows for its matched candidates.

    args = (transcript_path, spk, vid, seg, [(start, end, dur, xtx), ...])
    """
    path, spk, vid, seg, cands = args
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return []
    # index pipeline_v1 entries by rounded (start, end)
    entries = {}
    for e in doc.get("pipeline_v1", []):
        if "start" in e and "end" in e:
            entries[(round(float(e["start"]), 3), round(float(e["end"]), 3))] = e
    rows = []
    for start, end, dur, xtx in cands:
        e = entries.get((round(start, 3), round(end, 3)))
        if e is None:
            continue
        text = e.get(VOXTRAL)
        if not text or len(text.strip()) < 3:
            continue
        lang = e.get("whisper_language")
        prob = e.get("whisper_language_prob")
        if lang is None or prob is None:
            continue
        if not transcript_lang_ok(lang, text):
            continue
        rows.append((spk, vid, seg, start, end, dur, xtx, lang, float(prob),
                     " ".join(text.split())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtx-json", required=True,
                    help="big flat-dict JSON of segment xtx scores")
    ap.add_argument("--xtx-cache", default=None,
                    help="pre-extracted grep of voxblink2 xtx lines (skips the scan)")
    ap.add_argument("--transcripts-root", required=True,
                    help="root holding voxblink2/<...>/<seg>.json transcripts")
    ap.add_argument("--out", default="voxblink2_enriched.pkl")
    # Build-time pre-filter: looser than Stage A's gates (xtx>=0.85, 3-15s) so
    # Stage A still selects an identical set at its own thresholds.
    ap.add_argument("--min-xtx", type=float, default=0.80)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    args = ap.parse_args()

    # 1. stream xtx entries, pre-filter, group by transcript segment
    by_seg = collections.defaultdict(list)   # rel_head -> (spk,vid,seg,[cands])
    seen = kept = 0
    for key, xtx in iter_xtx(args.xtx_json, args.xtx_cache):
        seen += 1
        if xtx < args.min_xtx:
            continue
        parsed = parse_key(key)
        if parsed is None:
            continue
        head, spk, vid, seg, start, end = parsed
        dur = end - start
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        kept += 1
        by_seg[head].append((spk, vid, seg, start, end, dur, xtx))
        if seen % 2_000_000 == 0:
            print(f"  scanned {seen:,} xtx entries, kept {kept:,}", file=sys.stderr)
    print(f"xtx: scanned {seen:,}, kept {kept:,} candidates in "
          f"{len(by_seg):,} segments (xtx>={args.min_xtx}, "
          f"dur {args.min_dur}-{args.max_dur}s)", file=sys.stderr)

    # 2. read the surviving segments' transcripts (multiprocess) and join
    tasks = []
    for head, cands in by_seg.items():
        spk, vid, seg = cands[0][0], cands[0][1], cands[0][2]
        path = os.path.join(args.transcripts_root, head + ".json")
        tasks.append((path, spk, vid, seg,
                      [(c[3], c[4], c[5], c[6]) for c in cands]))

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(read_transcript, tasks, chunksize=256):
            rows.extend(res)
            done += 1
            if done % 20000 == 0:
                print(f"  joined {done:,}/{len(tasks):,} segments, "
                      f"{len(rows):,} rows", file=sys.stderr)

    # 3. summarize + write. lang_counts is a convenience (Stage A ignores it).
    lang_counts = collections.Counter(r[7] for r in rows)
    spk_per_lang = collections.defaultdict(set)
    for r in rows:
        spk_per_lang[r[7]].add(r[0])
    print(f"\nbuilt {len(rows):,} rows across {len(lang_counts)} languages",
          file=sys.stderr)
    for lang, c in lang_counts.most_common():
        print(f"  {lang}: {c:,} rows, {len(spk_per_lang[lang]):,} speakers",
              file=sys.stderr)

    with open(args.out, "wb") as f:
        pickle.dump((rows, dict(lang_counts)), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nwrote {args.out}  ({len(rows):,} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
