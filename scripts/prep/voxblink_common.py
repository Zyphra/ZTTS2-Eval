"""Shared paths of the VoxBlink2-derived in-the-wild data prep."""
import os

# Local VoxBlink2 audio tree (<spk>/<vid>/<seg>.wav). Set VOXBLINK2_AUDIO_ROOT to
# point at your copy of the dataset.
AUDIO_ROOT = os.environ.get("VOXBLINK2_AUDIO_ROOT", "/path/to/VoxBlink2/audio")
DATA_ROOT = "data/in-the-wild"


def wav_path(speaker, vid, seg):
    return f"{AUDIO_ROOT}/{speaker}/{vid}/{seg}.wav"
