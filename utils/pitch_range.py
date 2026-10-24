"""Register-normalized pitch range: the per-utterance F0 spread behind the pitch metric.

An utterance's pitch range is p90 - p10 of its voiced F0 in semitones relative to its OWN
voiced median, so it measures how much the melody moves and is blind to the speaker's
register (a low and a high voice with the same intonation have the same range).

    f0_track(path)    pYIN F0 (Hz, NaN = unvoiced) of the silence-trimmed utterance
    semitone_dev(f0)  semitones around the utterance median
    utt_range_st(st)  p90 - p10 of the voiced deviations
    range_st(path)    the range of one wav (None if unusable)
    ranges_cached     the same for many wavs, TSV-cached, process-pool parallel

pYIN is slow, so callers score a fixed uid sample (utils.benchmark.sample_uids).
"""
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SR = 16000
HOP = 320          # 20 ms, same grid as the rhythm views
FMIN, FMAX = 60.0, 400.0
MIN_SAMPLES = 8000  # trimmed audio shorter than 0.5 s is not scored
TRIM_DB = 35.0


def f0_track(path):
    """float32 F0 array (NaN = unvoiced) of the trimmed utterance, or None."""
    try:
        import librosa
        from scipy.signal import medfilt
        wav, _ = librosa.load(path, sr=SR)
        if wav.size == 0:
            return None
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
        return f0.astype(np.float32)
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
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                uid, _, val = line.rstrip("\n").partition("\t")
                if uid:
                    out[uid] = float(val) if val else None
    return out


def _write_tsv(path, data):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for uid in sorted(data):
            v = data[uid]
            fh.write(f"{uid}\t{'' if v is None else repr(float(v))}\n")
    os.replace(tmp, path)


def ranges_cached(wavs, cache_tsv, workers):
    """{uid: range in semitones or None} for `wavs` ({uid: wav path}).

    The cache is a TSV `uid<TAB>range` (empty = unusable wav). Only uids missing from it
    are computed; the result covers exactly the requested uids."""
    cache = _read_tsv(cache_tsv)
    todo = [u for u in wavs if u not in cache]
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(range_st, [wavs[u] for u in todo], chunksize=1))
        cache.update(dict(zip(todo, res)))
        _write_tsv(cache_tsv, cache)
    return {u: cache[u] for u in wavs}
