"""Generate TTS audio with CosyVoice3 (native upstream) for a ZTTS2-Eval test set.

Runs the official FunAudioLLM/CosyVoice code IN-PROCESS (cloned to
third_party/CosyVoice), not through vLLM-Omni -- the vllm-omni cosyvoice3 path was
too slow. Like generate_echo.py / generate_kokoro.py: a single device, sequential
loop. CosyVoice3 is a WITH-reference-text zero-shot cloner, so each utterance needs
the prompt wav AND its transcript (data/<task>/<lang>/prompt_text).

    ~/.venvs/cosyvoice/bin/python scripts/generate_cosyvoice.py \\
        --data-dir data/clean/en --out-dir results/cosyvoice3/clean/en
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
# The cloned upstream repo + its Matcha-TTS submodule must be importable.
_COSY = _REPO / "third_party" / "CosyVoice"
for p in (_COSY, _COSY / "third_party" / "Matcha-TTS"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, run_generate_loop)

# CosyVoice3 requires the prompt_text to carry a system prompt terminated by the
# special <|endofprompt|> token (it asserts token 151646 is present -- see
# third_party/CosyVoice/example.py): "<system>.<|endofprompt|>" + <prompt transcript>.
COSY3_SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"

# Default to the HF-cached snapshot (resolved dynamically) so offline compute nodes
# don't hit the network. Override with $COSYVOICE3_DIR or --model-dir.
def _default_model_dir():
    env = os.environ.get("COSYVOICE3_DIR")
    if env:
        return env
    import glob
    snaps = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/*")))
    # Fall back to the HF id (CosyVoice3 snapshot_downloads it) if nothing is cached.
    return snaps[-1] if snaps else "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


_DEFAULT_MODEL = _default_model_dir()


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with CosyVoice3 (native).")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text', 'prompt_wav.scp' and 'prompt_text'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--model-dir", default=_DEFAULT_MODEL,
                    help="CosyVoice3 model dir (HF id or local snapshot)")
    ap.add_argument("--fp16", action="store_true", help="load the model in fp16")
    ap.add_argument("--katakana", action="store_true",
                    help="convert the input text to katakana readings via pyopenjtalk "
                         "before synthesis -- CosyVoice3 REQUIRES katakana for Japanese "
                         "(its example.py: 'for Japanese usage, you must translate it to "
                         "katakana'); feeding raw kanji gives ~60%% WER")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking-rate multiplier")
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--stratified-bins", type=int, default=0,
                    help="if >0, pick K evenly-spaced utts across the full set")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    # CosyVoice3 needs the prompt transcript, so require all three.
    texts, prompts, prompt_texts, uids = load_test_set(args.data_dir)
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    to_kana = None
    if args.katakana:
        import pyopenjtalk  # noqa: E402
        to_kana = lambda t: pyopenjtalk.g2p(t, kana=True)  # noqa: E731
        print("[cosyvoice3] Japanese: converting text to katakana via pyopenjtalk", flush=True)

    print(f"[cosyvoice3] loading model from {args.model_dir} ...", flush=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402
    model = CosyVoice3(args.model_dir, load_vllm=False, fp16=args.fp16)
    sr = model.sample_rate

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            # CosyVoice's frontend loads/resamples the prompt itself, so pass the PATH
            # (its extractors call load_wav internally -- a pre-loaded tensor 400s with
            # "Invalid file: tensor(...)").
            tts_text = to_kana(texts[uid]) if to_kana else texts[uid]
            chunks = [j["tts_speech"] for j in model.inference_zero_shot(
                tts_text, COSY3_SYS_PROMPT + prompt_texts[uid], prompts[uid],
                stream=False, speed=args.speed)]
            if not chunks:
                return uid, "empty audio", 0.0
            wav = torch.concat(chunks, dim=1).cpu()      # [1, T] at model.sample_rate
            if wav.numel() == 0:
                return uid, "empty audio", 0.0
            tmp = out_path + ".tmp.wav"
            torchaudio.save(tmp, wav, sr)
            os.replace(tmp, out_path)                     # atomic
            return uid, "ok", wav.shape[-1] / sr
        except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
            # format_error logs the TYPE too: several exceptions here stringify
            # to "" (61/403 of in-the-wild/ko failed that way), which left the log
            # reading "[gen] uttid_N: error: " with nothing to diagnose.
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
