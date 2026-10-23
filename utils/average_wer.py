import json
import os
import sys

import numpy as np

# 95% CI on the per-utterance mean via a seeded percentile bootstrap. Per-utt WER
# is bounded-below, skewed and heavy-tailed (a few >1.0 utts), so a normal-approx
# SE understates the interval; the bootstrap needs no distributional assumption.
# The seed is fixed so re-aggregating the same .raw file is byte-reproducible.
# This module is import-safe: utils/score_msr_utmos.py imports boot_ci95/CI_BOOT/
# CI_SEED from here (the canonical copy) instead of keeping its own.
CI_BOOT = 10000
CI_SEED = 20260717

# Minimum scored/expected coverage before the aggregate is trusted. Below it the
# .wer is still written in full (so the partial artifact stays inspectable) but
# the exit code is non-zero, so cal_wer.sh / resume gates see the failure instead
# of a success-looking stub (the 3-of-1279 incident rescore_stale_en.sh documents).
MIN_COVERAGE = float(os.environ.get("SCORE_MIN_COVERAGE", "0.98"))

# Rows whose hypothesis is this sentinel were not transcribed at all: the reference
# ASR could not read the generated audio, so utils/run_wer.py scored the utterance
# as maximum error instead of dropping it (see the long note there). They are real
# rows and belong in the mean -- excluding them is the survivor bias the imputation
# exists to remove -- but the footer reports them separately so a leaf that is
# largely imputed can never be mistaken for a leaf that was largely transcribed.
UNSCOREABLE_TAG = "<UNSCOREABLE"
# Warn (not fail) above this imputed share; SCORE_MAX_IMPUTED turns it into a gate.
MAX_IMPUTED = float(os.environ.get("SCORE_MAX_IMPUTED", "1.0"))
IMPUTED_WARN = 0.02


def manifest_for(raw_path):
    """The scorer's input list for this .raw, or None.

    ``<leaf>/wav_res_ref_text.wer.raw`` and the control variants
    (``…whisperctl.raw``) all derive from ``<leaf>/wav_res_ref_text``, so the stem
    before the first dot names the manifest. Used to recover n_expected for a .raw
    written before the .meta sidecar existed, where assuming n_expected == n_scored
    made a leaf that scored 169 of 1500 utterances report full coverage.
    """
    d, b = os.path.split(raw_path)
    cand = os.path.join(d, b.split(".", 1)[0])
    return cand if os.path.isfile(cand) else None


def boot_ci95(vals, scale=1.0, B=CI_BOOT, seed=CI_SEED, ndigits=3):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # resample in row-chunks to cap peak memory on large n (chunked draws consume
    # the generator stream identically to one (B, n) draw, so results are exact)
    step = max(1, 2_000_000 // max(a.size, 1))
    out = []
    done = 0
    while done < B:
        m = min(step, B - done)
        idx = rng.integers(0, a.size, size=(m, a.size))
        out.append(a[idx].mean(axis=1))
        done += m
    means = np.concatenate(out) * scale
    return (round(float(np.percentile(means, 2.5)), ndigits),
            round(float(np.percentile(means, 97.5)), ndigits))


def main(infile, outfile):
    fout = open(outfile, "w")
    fout.write("utt" + '\t' + "wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\t' + 'res_wer_ins' + '\t' + 'res_wer_del' + '\t' + 'res_wer_sub' + '\n')
    wers = []
    wers_below50 = []
    wers_clip = []
    inses = []
    deles = []
    subses = []
    n_higher_than_50 = 0
    n_imputed = 0
    with open(infile, "r") as fin:
        for lineno, line in enumerate(fin, 1):
            try:
                wav_path, wer, text_ref, text_res, inse, dele, subs = line.rstrip("\n").split("\t")
                wer_f, inse_f, dele_f, subs_f = float(wer), float(inse), float(dele), float(subs)
            except ValueError as e:
                sys.exit(f"{infile}:{lineno}: malformed line ({e})")
            if text_res.startswith(UNSCOREABLE_TAG):
                n_imputed += 1
            if wer_f > 0.5:
                n_higher_than_50 += 1
            else:
                wers_below50.append(wer_f)

            wers.append(wer_f)
            wers_clip.append(min(wer_f, 1.0))
            inses.append(inse_f)
            deles.append(dele_f)
            subses.append(subs_f)
            fout.write(line)

    if not wers:
        sys.exit(f"error: no scored utterances in {infile}")

    wer = round(np.mean(wers)*100,3)
    # all-utterances-above-50% edge case: report L50 over the full set rather than NaN
    wer_below50 = round(np.mean(wers_below50)*100,3) if wers_below50 else wer
    wer_clip = round(np.mean(wers_clip)*100,3)
    subs = round(np.mean(subses)*100,3)
    dele = round(np.mean(deles)*100,3)
    inse = round(np.mean(inses)*100,3)

    fout.write(f"WER: {wer}, WER_L50: {wer_below50}, WER_CLIP: {wer_clip}\n")
    # 95% bootstrap CI, on a separate line so the strict WER: parsers above are untouched.
    wer_lo, wer_hi = boot_ci95(wers, 100.0)
    clip_lo, clip_hi = boot_ci95(wers_clip, 100.0)
    fout.write(f"WER_CI95: [{wer_lo}, {wer_hi}], WER_CLIP_CI95: [{clip_lo}, {clip_hi}], "
               f"n: {len(wers)}, boot: {CI_BOOT}, seed: {CI_SEED}\n")

    # Coverage footer -- strictly additive lines (the WER:/WER_CI95: formats above
    # are unchanged). n_expected comes from run_wer.py's `<raw>.meta` sidecar (the
    # wav list the scorer was given); a legacy .raw without one reports its own
    # row count, i.e. claims full coverage like before.
    n_scored = len(wers)
    n_expected = None
    meta_path = infile + ".meta"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                n_expected = int(json.load(f)["n_expected"])
        except (OSError, ValueError, KeyError) as e:
            print(f"warning: unreadable {meta_path}: {e}", file=sys.stderr)
    if n_expected is None:
        # No sidecar (a .raw from before it existed): count the scorer's input list
        # rather than assuming the run was complete.
        man = manifest_for(infile)
        if man:
            with open(man, encoding="utf-8", errors="replace") as f:
                n_expected = sum(1 for ln in f if ln.strip())
        else:
            print(f"warning: no .meta and no manifest for {infile}; "
                  "coverage assumes the run was complete", file=sys.stderr)
            n_expected = n_scored
    fout.write(f"N_SCORED: {n_scored}\n")
    fout.write(f"N_EXPECTED: {n_expected}\n")
    # Additive: how much of the mean is measurement and how much is maximum-error
    # imputation for audio the reference ASR could not transcribe at all.
    fout.write(f"N_TRANSCRIBED: {n_scored - n_imputed}\n")
    fout.write(f"N_IMPUTED: {n_imputed}\n")
    fout.close()

    print(f"WER: {wer}%\n")
    print(n_higher_than_50)

    imp_frac = n_imputed / n_scored if n_scored else 0.0
    if imp_frac > IMPUTED_WARN:
        print(f"warning: {n_imputed}/{n_scored} = {imp_frac:.3f} of {outfile} is "
              "maximum-error imputation for unscoreable audio, not transcription",
              file=sys.stderr)
    if imp_frac > MAX_IMPUTED:
        sys.exit(f"error: imputed share {imp_frac:.3f} > {MAX_IMPUTED} "
                 f"(SCORE_MAX_IMPUTED) -- {outfile} written")

    if n_expected > 0 and n_scored / n_expected < MIN_COVERAGE:
        sys.exit(f"error: WER coverage {n_scored}/{n_expected} = "
                 f"{n_scored / n_expected:.3f} < {MIN_COVERAGE} (SCORE_MIN_COVERAGE) "
                 f"-- {outfile} written but the leaf is not trustworthy")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
