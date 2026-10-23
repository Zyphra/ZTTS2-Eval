"""Generate TTS audio with VibeVoice-Realtime-0.5B (Microsoft) for a ZTTS2-Eval set.

VibeVoice-Realtime-0.5B is a ~0.5B streaming TTS. It is a PRESET-voice model (fixed
speaker embeddings, e.g. en-Carter_man), NOT a zero-shot cloner -- it ignores the
reference/prompt wav (speaker-similarity is not meaningful; WER + MSR-UTMOS are).
Runs in-process (no server), like generate_neutts.py, adapting the upstream
demo/realtime_model_inference_from_file.py.

Languages: primarily English, but upstream ships preset voices for nine more
(de fr it jp kr nl pl pt sp) -- select with --voice-lang, which names the VOICE FILE
prefix, not our language code (run_generate_vibevoice.sh maps ja->jp, ko->kr, es->sp).
The model card ships those voices while its limitations section still says "trained
only on English data", so non-English output is vendor-flagged exploratory.

    ~/.venvs/vibevoice/bin/python scripts/generate_vibevoice.py \\
        --data-dir data/zero_shot/en --out-dir results/vibevoice-rt-0.5b/zero_shot/en
"""
import argparse
import copy
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
_VV = _REPO / "third_party" / "VibeVoice"
if _VV.is_dir() and str(_VV) not in sys.path:
    sys.path.insert(0, str(_VV))
from utils.genclient import (  # noqa: E402
    apply_limit, format_error, pick_index, run_generate_loop)
from utils.scp import load_scp  # noqa: E402

MODEL_ID = os.environ.get("VIBEVOICE_DIR", "microsoft/VibeVoice-Realtime-0.5B")
VOICE_DIR = os.environ.get("VIBEVOICE_VOICE_DIR", str(_VV / "demo" / "voices" / "streaming_model"))


def main():
    ap = argparse.ArgumentParser(description="Generate TTS audio with VibeVoice-Realtime-0.5B.")
    ap.add_argument("--data-dir", required=True,
                    help="test-set dir with 'text' (prompt_wav IGNORED -- preset voice)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=MODEL_ID)
    ap.add_argument("--voice-lang", default="en",
                    help="preset-voice file prefix in VOICE_DIR: en|de|fr|it|jp|kr|nl|pl|pt|sp "
                         "(NOT our lang code -- ja->jp, ko->kr, es->sp)")
    ap.add_argument("--speaker", default="rotate",
                    help="fixed speaker within --voice-lang (en: Carter|Davis|Emma|Frank|"
                         "Grace|Mike; others: Spk0|Spk1) or 'rotate' (deterministic "
                         "per-utterance rotation)")
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratified-bins", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    texts = load_scp(os.path.join(args.data_dir, "text"))
    uids = list(texts.keys())            # preset voice -> every utterance in `text`
    uids = apply_limit(uids, args.limit, args.stratified_bins)

    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[vibevoice] loading {args.model_dir} on {device} ...", flush=True)
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_dir)
    # flash_attention_2 is the upstream default but isn't installed here -> sdpa.
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_dir, torch_dtype=dtype, device_map=device, attn_implementation="sdpa")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    # Preset voice prompts: pre-load this language's speakers once; rotate per utterance.
    all_pt = sorted(glob.glob(os.path.join(VOICE_DIR, f"{args.voice_lang}-*.pt")))
    if not all_pt:
        have = sorted({os.path.basename(p).split("-", 1)[0]
                       for p in glob.glob(os.path.join(VOICE_DIR, "*-*.pt"))})
        raise SystemExit(f"no preset voices for --voice-lang '{args.voice_lang}' in "
                         f"{VOICE_DIR} (have: {' '.join(have) or '<none>'})")
    names = [os.path.basename(p).split("-", 1)[1].split("_", 1)[0] for p in all_pt]  # Carter,...
    if args.speaker != "rotate":
        sel = [i for i, n in enumerate(names) if n.lower() == args.speaker.lower()]
        if not sel:
            raise SystemExit(f"unknown speaker '{args.speaker}' for voice-lang "
                             f"'{args.voice_lang}' (have {names})")
        all_pt = [all_pt[sel[0]]]; names = [names[sel[0]]]
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prompts = [torch.load(p, map_location=device, weights_only=True) for p in all_pt]
    print(f"[vibevoice] voice-lang={args.voice_lang} speakers: {names}", flush=True)

    def speaker_idx(uid):
        # uid-hash rotation (not enumeration index) so a resume with different
        # --limit/--stratified-bins never reassigns voices mid-set. NOTE: this
        # reassigns voices once vs runs generated before the hash switch.
        return pick_index(uid, len(prompts))

    # voice manifest -- deterministic, so write it upfront (recoverable later).
    vpath = os.path.join(args.out_dir, "voices.txt")
    with open(vpath, "w", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\t{names[speaker_idx(uid)]}\n")
    print(f"[vibevoice] voice assignments -> {vpath}", flush=True)

    def synth(text, prompt):
        inputs = processor.process_input_with_cached_prompt(
            text=text, cached_prompt=prompt, padding=True,
            return_tensors="pt", return_attention_mask=True)
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer, generation_config={"do_sample": False},
            verbose=False, all_prefilled_outputs=copy.deepcopy(prompt))
        so = outputs.speech_outputs
        return so[0] if so and so[0] is not None else None

    def generate(uid):
        out_path = os.path.join(wav_dir, uid + ".wav")
        if os.path.exists(out_path) and not args.overwrite:
            return uid, "skip", 0.0
        try:
            audio = synth(texts[uid], prompts[speaker_idx(uid)])
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
        except Exception as e:  # noqa: BLE001
            return uid, format_error(e), 0.0

    # workers=1: in-process GPU engine -> sequential loop in this thread.
    run_generate_loop(uids, generate, workers=1, desc="generate", wav_dir=wav_dir)


if __name__ == "__main__":
    main()
