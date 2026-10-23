"""Shared parsers for the per-utterance score files written into every results leaf.

A scored leaf (``results/<system>/<task>/<lang>/``) carries three uid-keyed score
files, each in its own format:

    wav_res_ref_text.wer.raw   <abs_wav_path> TAB wer TAB ref TAB hyp TAB ins TAB del TAB sub
    spk_simi_scores.txt        <uttid> SP <ref_score> SP <hyp_score>      (+ trailing `avg` row)
    msr_utmos_scores.txt       <uttid> TAB <wav_path> TAB <score>

The uid is the join key across all three, and is stable across systems for a given
(task, lang). Note the WER file keys on ``basename(wav_path)`` rather than carrying
the uid directly -- the ``utt`` column its header advertises is never emitted.

Every parser is defensive: a missing or unreadable file yields an empty dict rather
than raising, because leaves are routinely partial (a metric that has not backfilled
yet, a language a system does not support). Callers must therefore check for
emptiness, not rely on an exception.

Stdlib only -- ``explorer/serve.py`` is a zero-dependency server and imports from here.
"""

import os

# Aggregation constants, mirrored from utils/average_wer.py so subset re-aggregation
# is byte-comparable with the pipeline's own numbers.
CI_BOOT = 10000
CI_SEED = 20260717
# average_wer.py buckets an utterance into WER_L50 when `wer <= 0.5` (it tests
# `wer_f > 0.5` and puts the failures in the other bin).
L50_THRESHOLD = 0.5


def read_lines(path):
    """File contents as a list of lines, or ``[]`` if it cannot be read."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def parse_scp(path):
    """``<uid> <path>`` -> ``{uid: path}``. Used for wav.scp / prompt_wav.scp.

    Collapses the doubled slash that ``data/zero_shot/`` prompt paths carry
    (``data/zero_shot//en/waveform/...``).
    """
    out = {}
    for line in read_lines(path):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip().replace("//", "/")
    return out


def parse_text(path):
    """``<uid> <free text>`` -> ``{uid: text}``. Used for text / prompt_text."""
    out = {}
    for line in read_lines(path):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
        elif len(parts) == 1:
            out[parts[0]] = ""
    return out


def parse_wer(path):
    """Parse ``wav_res_ref_text.wer[.raw]`` -> ``{uid: {wer, ref_text, hyp_text, ins, del, sub}}``.

    Handles both variants: ``.raw`` is headerless data rows only, ``.wer`` prepends a
    header row and appends two ``WER:``/``WER_CI95:`` footer lines. The header is
    dropped by its ``utt`` first field and the footers by their column count, so the
    same parser reads either file.

    ``wer`` is the raw per-utterance fraction (can exceed 1.0); ins/del/sub are edit
    counts divided by the reference token count.
    """
    out = {}
    for line in read_lines(path):
        if not line:
            continue
        c = line.split("\t")
        if c[0] == "utt":  # header row of the .wer variant
            continue
        if len(c) < 4:  # WER:/WER_CI95: footer lines
            continue
        uid = os.path.splitext(os.path.basename(c[0]))[0]

        def num(i):
            try:
                return float(c[i])
            except (ValueError, IndexError):
                return None

        out[uid] = {
            "wer": num(1),
            "ref_text": c[2] if len(c) > 2 else None,
            "hyp_text": c[3] if len(c) > 3 else None,
            "ins": num(4),
            "del": num(5),
            "sub": num(6),
        }
    return out


def parse_spk(path):
    """``<uid> <ref_score> <hyp_score>`` -> ``{uid: hyp_score}`` (cosine x100).

    Takes the LAST field: column 2 is the reference-vs-reference score, which is
    ``0.00`` unless the scorer ran with ``--ref_wavs``. The trailing ``avg`` row is
    skipped.
    """
    out = {}
    for line in read_lines(path):
        parts = line.split()
        if len(parts) < 3 or parts[0] == "avg":
            continue
        try:
            out[parts[0]] = float(parts[-1])
        except ValueError:
            pass
    return out


def parse_score_path(path):
    """``<uid> TAB <wav_path> TAB <score>`` -> ``{uid: score}``. Used for msr_utmos."""
    out = {}
    for line in read_lines(path):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                out[parts[0]] = float(parts[-1])
            except ValueError:
                pass
    return out


def wer_clip(wer):
    """Per-utterance WER_CLIP in percent: cap at 100% then scale, per average_wer.py:54."""
    return min(wer, 1.0) * 100.0


def load_leaf(leaf_dir):
    """Load the per-uid scores of one results leaf.

    Returns ``{"wer": {uid: raw_fraction}, "wer_clip": {uid: pct}, "spk": {uid: x100},
    "msr": {uid: mos}, "detail": {uid: {...wer fields...}}}``. Any metric whose file is
    missing comes back as an empty dict -- partial leaves are normal, so check
    emptiness rather than trusting that all four are populated.

    Prefers ``.wer.raw`` over ``.wer``: it is the scorer's direct output, whereas the
    ``.wer`` copy is rewritten by the aggregation step.
    """
    raw = os.path.join(leaf_dir, "wav_res_ref_text.wer.raw")
    if not os.path.isfile(raw):
        raw = os.path.join(leaf_dir, "wav_res_ref_text.wer")
    detail = parse_wer(raw)
    wer = {u: d["wer"] for u, d in detail.items() if d["wer"] is not None}
    return {
        "wer": wer,
        "wer_clip": {u: wer_clip(v) for u, v in wer.items()},
        "spk": parse_spk(os.path.join(leaf_dir, "spk_simi_scores.txt")),
        "msr": parse_score_path(os.path.join(leaf_dir, "msr_utmos_scores.txt")),
        "detail": detail,
    }
