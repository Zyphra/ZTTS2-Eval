"""Generate TTS audio with VibeVoice-1.5B (Microsoft) for a ZTTS2-Eval set.

Unlike VibeVoice-Realtime-0.5B (scripts/gen/generate_vibevoice.py), which is a PRESET-voice
model, VibeVoice-1.5B is a genuine ZERO-SHOT CLONER: VibeVoiceProcessor takes
`voice_samples`, so we feed it each utterance's reference wav from prompt_wav.scp and
speaker-similarity IS meaningful for this model.

The TTS inference code was withdrawn from upstream ("Installation and Usage: Disabled
due to widespread misuse") and does not exist in any reachable commit, so it is vendored
under third_party/VibeVoice-1.5B -- see that tree's PROVENANCE.md for the source, the
diff-against-official audit and the security review. That tree must be on PYTHONPATH,
and must NOT be mixed with third_party/VibeVoice (the realtime 0.5B needs a different
version of the same module names).

Languages: English and Chinese only (HF tags `en`, `zh`; the card's limitations say
"Transcripts in languages other than English or Chinese may result in unexpected audio
outputs").

    ~/.venvs/vibevoice15/bin/python scripts/gen/generate_vibevoice15.py \\
        --data-dir data/clean/en --out-dir results/vibevoice-1.5b/clean/en
"""
import argparse
import os
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
_VV15 = Path(REPO) / "third_party" / "VibeVoice-1.5B"
if _VV15.is_dir() and str(_VV15) not in sys.path:
    sys.path.insert(0, str(_VV15))
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, load_test_set, run_generate_loop)

MODEL_ID = os.environ.get("VIBEVOICE15_DIR", "microsoft/VibeVoice-1.5B")


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with VibeVoice-1.5B.")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text' and 'prompt_wav.scp' (clone mode)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=MODEL_ID)
    # Upstream demo/inference_from_file.py defaults: cfg_scale 1.3, 10 DDPM steps.
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    ap.add_argument("--ddpm-steps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified-bins", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    texts, prompt_wavs, _, uids = load_test_set(args.data_dir,
                                                need_prompt_text=False)
    missing = len(texts) - len(uids)
    if missing:
        print(f"[vibevoice15] {missing} utterances have no prompt_wav -- skipped", flush=True)
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    import torch
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[vibevoice15] loading {args.model_dir} on {device} ...", flush=True)
    processor = VibeVoiceProcessor.from_pretrained(args.model_dir)
    # flash_attention_2 is the upstream default but isn't installed here -> sdpa,
    # same choice as the realtime 0.5B recipe.
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_dir, torch_dtype=dtype, device_map=device, attn_implementation="sdpa")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    def synth(text, ref_wav):
        # VibeVoice is script-formatted and multi-speaker; our sets are single-utterance,
        # so every item is a one-turn "Speaker 1:" script with one reference voice.
        inputs = processor(
            text=[f"Speaker 1: {text}"],
            voice_samples=[[ref_wav]],
            padding=True, return_tensors="pt", return_attention_mask=True)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer, generation_config={"do_sample": False},
            verbose=False)
        so = outputs.speech_outputs
        return so[0] if so and so[0] is not None else None

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            audio = synth(texts[uid], prompt_wavs[uid])
            if audio is None:
                return uid, "empty audio", 0.0
            tmp = out_path + ".tmp.wav"
            processor.save_audio(audio, output_path=tmp, sampling_rate=24000)
            os.replace(tmp, out_path)                     # atomic
            try:
                dur = float(audio.numel()) / 24000.0
            except Exception:  # noqa: BLE001 - duration is informational only
                dur = 0.0
            return uid, "ok", dur
        except Exception as e:  # noqa: BLE001 -- log type AND message, then continue
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
