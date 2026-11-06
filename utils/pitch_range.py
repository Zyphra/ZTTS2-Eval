"""Register-normalized pitch range: the per-utterance F0 spread behind the pitch metric.

An utterance's pitch range is p90 - p10 of its voiced F0 in semitones relative to its OWN
voiced median, so it measures how much the melody moves and is blind to the speaker's
register (a low and a high voice with the same intonation have the same range).

    f0_of(wav)        pYIN F0 (Hz, NaN = unvoiced) of a 16 kHz array, non-speech removed
    f0_track(path)    the same for a wav file
    semitone_dev(f0)  semitones around the utterance median
    utt_range_st(st)  p90 - p10 of the voiced deviations
    range_st(path)    the range of one wav (None if unusable)
    ranges_cached     the same for many wavs, TSV-cached, process-pool parallel

pYIN reports pitch where there is no voice, and those frames sit far from the speaker's
F0, so they dominate p90 - p10. Two steps remove them (docs/controls_2026-08.md C10):
voiced frames more than 50 dB below the loudest frame (near-silent pauses, which pYIN
pins at its 400 Hz ceiling), and voiced runs whose median lies 9 st or more from the
utterance's voiced median (background hum or music bass far below the voice, ceiling
pins far above it). With both, pYIN and PENN rank the Clean systems alike.

pYIN is slow, so callers score a fixed uid sample (utils.benchmark.sample_uids).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from utils.atomic import write_text_atomic

SR = 16000
HOP = 320          # 20 ms, same grid as the rhythm views
FMIN, FMAX = 60.0, 400.0
MIN_SAMPLES = 8000  # trimmed audio shorter than 0.5 s is not scored
TRIM_DB = 35.0
GATE_DB = -50.0     # voiced frames this far below the loudest frame are not voice
RUN_MAX_ST = 9.0    # voiced runs whose median is this far from the utterance median are not voice
# Stamped into every range cache: entries computed by another method are recomputed.
METHOD = f"pyin {FMIN:g}-{FMAX:g} trim{TRIM_DB:g} gate{GATE_DB:g} run{RUN_MAX_ST:g}"


def drop_outlier_runs(f0, max_st=RUN_MAX_ST):
    """NaN out each run of consecutive voiced frames whose median F0 lies max_st or more
    from the utterance's voiced median. A real excursion inside a voiced run survives;
    a separate run an octave off the voice does not."""
    voiced = np.isfinite(f0)
    if voiced.sum() < 10:
        return f0
    med = np.median(f0[voiced])
    edges = np.flatnonzero(np.diff(np.concatenate(([0], voiced.astype(np.int8), [0]))))
    out = f0.copy()
    for s, e in zip(edges[::2], edges[1::2]):
        if abs(12.0 * np.log2(np.median(f0[s:e]) / med)) >= max_st:
            out[s:e] = np.nan
    return out


def f0_of(wav):
    """float32 F0 array (NaN = unvoiced) of a 16 kHz utterance after edge trimming, or None."""
    import librosa
    from scipy.signal import medfilt
    wav, _ = librosa.effects.trim(wav, top_db=TRIM_DB)
    if len(wav) < MIN_SAMPLES:
        return None
    f0, _, _ = librosa.pyin(wav, fmin=FMIN, fmax=FMAX, sr=SR,
                            frame_length=1024, hop_length=HOP)
    if f0 is None or np.all(np.isnan(f0)):
        return None
    # width-3 median filter on voiced runs kills single-frame octave blips
    v = np.where(np.isnan(f0), 0.0, f0)
    vm = medfilt(v, 3)
    f0 = np.where(np.isnan(f0), np.nan, np.where(vm > 0, vm, f0))
    # same 1024/320 centered frames as pyin, so frame i is the same audio
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=HOP)[0][:len(f0)]
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / max(float(rms.max()), 1e-12))
    f0 = np.where(db >= GATE_DB, f0, np.nan)
    return drop_outlier_runs(f0).astype(np.float32)


def f0_track(path):
    """f0_of a wav file, or None if unreadable or unusable."""
    try:
        import librosa
        wav, _ = librosa.load(path, sr=SR)
        if wav.size == 0:
            return None
        return f0_of(wav)
    except Exception:
        return None


def semitone_dev(f0):
    """Semitones relative to the utterance's own voiced median (NaN unvoiced)."""
    voiced = f0[np.isfinite(f0)]
    if voiced.size < 10:
        return None
    return 12.0 * np.log2(f0 / np.median(voiced))


def utt_range_st(st):
    """Pitch range of one utterance: p90 - p10 of its voiced semitone deviations."""
    v = st[np.isfinite(st)]
    return float(np.percentile(v, 90) - np.percentile(v, 10))


def range_st(path):
    f0 = f0_track(path)
    if f0 is None:
        return None
    st = semitone_dev(f0)
    return None if st is None else utt_range_st(st)


def _read_tsv(path):
    """Cached ranges, or {} when the file is missing or was written by another METHOD."""
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            if fh.readline().rstrip("\n") != f"#method\t{METHOD}":
                return {}
            for line in fh:
                uid, _, val = line.rstrip("\n").partition("\t")
                if uid:
                    out[uid] = float(val) if val else None
    return out


def _write_tsv(path, data):
    lines = [f"#method\t{METHOD}\n"]
    for uid in sorted(data):
        v = data[uid]
        lines.append(f"{uid}\t{'' if v is None else repr(float(v))}\n")
    write_text_atomic(path, "".join(lines))


def ranges_cached(wavs, cache_tsv, workers):
    """{uid: range in semitones or None} for `wavs` ({uid: wav path}).

    The cache is a TSV `uid<TAB>range` (empty = unusable wav) under a `#method` line;
    a cache from another method is discarded. Only uids missing from it are computed;
    the result covers exactly the requested uids."""
    cache = _read_tsv(cache_tsv)
    todo = [u for u in wavs if u not in cache]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(range_st, [wavs[u] for u in todo], chunksize=1))
        cache.update(dict(zip(todo, res)))
        try:
            _write_tsv(cache_tsv, cache)
        except OSError as e:                       # read-only tree: still correct, just uncached
            print(f"[warn] pitch cache not written: {e}")
    return {u: cache[u] for u in wavs}
