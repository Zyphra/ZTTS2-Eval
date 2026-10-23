"""Generate TTS audio with MOSS-TTS (native upstream) for a ZTTS2-Eval test set.

Runs the official OpenMOSS/MOSS-TTS code IN-PROCESS via transformers (AutoModel +
AutoProcessor, trust_remote_code) -- NOT vLLM-Omni, whose moss_tts codec
reimplementation is weight-incompatible with the published MOSS-Audio-Tokenizer
checkpoint. The HF processor carries its OWN audio_tokenizer, which matches the
checkpoint. MossTTSDelay is an 8B zero-shot cloner: clone mode conditions on the
reference wav's timbre (audio only, no transcript needed).

    ~/.venvs/moss-tts/bin/python scripts/generate_moss_tts.py \\
        --data-dir data/zero_shot/en --out-dir results/moss-tts/zero_shot/en
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, run_generate_loop,
    save_wav_atomic)

MODEL_ID = os.environ.get("MOSS_TTS_DIR", "OpenMOSS-Team/MOSS-TTS")


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with MOSS-TTS (native).")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text' and 'prompt_wav.scp'")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; wavs are written to <out-dir>/wavs/")
    ap.add_argument("--model-dir", default=MODEL_ID, help="MOSS-TTS model (HF id or path)")
    ap.add_argument("--attn", default="",
                    help="override attn_implementation (e.g. sdpa) -- MOSS-TTS-Local-v1.5's "
                         "config requests flash_attention_2, which isn't installed; empty "
                         "= use the model's config default")
    # Sampling defaults follow the MOSS-TTS reference app.
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="only the first N utterances")
    ap.add_argument("--stratified-bins", type=int, default=0,
                    help="if >0, pick K evenly-spaced utts across the full set")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if the output wav already exists")
    args = ap.parse_args()

    texts, prompts, _, uids = load_test_set(args.data_dir, need_prompt_text=False)
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[moss-tts] loading {args.model_dir} on {device} ...", flush=True)
    from transformers import AutoModel, AutoProcessor  # noqa: E402
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    mkw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.attn:
        mkw["attn_implementation"] = args.attn
    model = AutoModel.from_pretrained(args.model_dir, **mkw).to(device)
    model.eval()
    # sample rate: processors expose it via .model_config.sampling_rate, but some
    # variants (e.g. Nano) use a bare tokenizer without model_config -- fall back to
    # the model config, then 24 kHz (48 kHz for the v2-tokenizer stereo variants).
    mc = getattr(processor, "model_config", None)
    sr = (getattr(mc, "sampling_rate", None)
          or getattr(getattr(model, "config", None), "sampling_rate", None)
          or 24000)
    sr = int(sr)

    def synth(uid):
        # Clone mode: timbre from the reference wav (audio only, no transcript).
        conv = [[processor.build_user_message(text=texts[uid], reference=[prompts[uid]])]]
        batch = processor(conv, mode="generation")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                audio_temperature=args.temperature,
                audio_top_p=args.top_p,
                audio_top_k=args.top_k,
                audio_repetition_penalty=args.repetition_penalty,
            )
        messages = processor.decode(outputs)
        if not messages or messages[0] is None:
            return None
        audio = messages[0].audio_codes_list[0]
        audio = (audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor)
                 else np.asarray(audio, dtype=np.float32))
        return audio.reshape(-1)

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            wav = synth(uid)
            if wav is None or wav.size == 0:
                return uid, "empty audio", 0.0
            dur = save_wav_atomic(wav, sr, out_path)
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001 -- log & continue, mirror other clients
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
