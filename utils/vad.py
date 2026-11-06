"""Speech segments from the pinned Silero VAD (utils.benchmark.SILERO_VAD), for metrics that
must tell speech from non-speech regardless of level: the paired pause count
(scripts/prosody/paired_itw.py) and the rhythm edges (scripts/prosody/rhythm_dist.py). An energy rule
relative to the peak cannot: background noise within 35 dB of the peak hides pauses, and a
generated lead-in tone or noise burst reads as speech (docs/controls_2026-08.md C8, C9)."""
import numpy as np

from utils.benchmark import SILERO_VAD

SR = 16000
# Silero's own default speech pad: its segment onsets sit slightly after the acoustic onset.
EDGE_PAD_S = 0.03
# Identifies the edge trim in caches/keys: pinned Silero revision + pad.
TRIM_TAG = f"silero:{SILERO_VAD[1][:7]}:pad{EDGE_PAD_S:g}"

_VAD = None


def load():
    """(model, get_speech_timestamps), loaded once per process. Call it before a worker pool
    forks so the workers inherit the model; it pins torch to one thread per process."""
    global _VAD
    if _VAD is None:
        import torch
        torch.set_num_threads(1)
        model, utils = torch.hub.load(f"{SILERO_VAD[0]}:{SILERO_VAD[1]}", "silero_vad",
                                      trust_repo=True, skip_validation=True, verbose=False)
        _VAD = (model, utils[0])
    return _VAD


def speech_segments(wav, min_silence_s=0.2):
    """[(start, end)] sample indices of the speech in a 16 kHz waveform, at Silero's default
    thresholds with no padding. The VAD sees the peak-normalized signal: Silero's speech
    probability depends on level (a -20 dB gain alone moved the paired pause count from
    0.62 to 0.86 pauses/utt)."""
    if wav.size == 0 or not np.any(wav):
        return []
    import torch
    model, speech_timestamps = load()
    ts = speech_timestamps(torch.from_numpy((wav / np.abs(wav).max()).astype(np.float32)), model,
                           sampling_rate=SR, min_silence_duration_ms=int(min_silence_s * 1000),
                           speech_pad_ms=0)
    return [(t["start"], t["end"]) for t in ts]


def speech_span(wav, pad_s=EDGE_PAD_S):
    """(start, end) sample indices from the first speech onset to the last speech offset,
    padded by pad_s and clipped to the file, or None when the VAD finds no speech."""
    segs = speech_segments(wav)
    if not segs:
        return None
    pad = int(pad_s * SR)
    return max(0, segs[0][0] - pad), min(len(wav), segs[-1][1] + pad)


def trim_speech(wav16, pad_s=EDGE_PAD_S):
    """wav16 cut to speech_span (16 kHz), or None when the VAD finds no speech."""
    span = speech_span(wav16, pad_s)
    return None if span is None else wav16[span[0]:span[1]]
