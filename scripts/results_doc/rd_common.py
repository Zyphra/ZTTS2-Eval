#!/usr/bin/env python3
"""Shared constants + helpers for the results_doc builders and plots.

One place for the things that used to drift as per-file copies:

  - the model display-label map (`MODEL_LABEL` / `label()`), previously
    diverging variants across gen_md.py, build_mpm_dist.py,
    build_reliability.py, and the scripts/plot_* files;
  - system discovery from the results tree (`discover_systems()`):
    hard-coded model lists are how caches came to silently hold only stale
    baselines;
  - the language display order (`ordered_langs()`), derived from the frozen
    baskets in utils/benchmark.py, previously seven diverging hard-coded lists;
  - atomic JSON writes (`write_json_atomic()`), so a crash mid-dump can never
    leave a truncated cache/snapshot file for the explorer to choke on.
"""
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")

sys.path.insert(0, REPO_ROOT)
from utils.benchmark import CLEAN_LANGS, ITW_LANGS  # noqa: E402

# --- model labels -----------------------------------------------------------
# Optional pretty labels for well-known public models; any other model is shown
# with its results/ directory name minus the omni_ engine prefix. The zonos2_
# prefix is deliberately NOT stripped: the variant name alone ("accurate",
# "expressive", ...) is meaningless out of context. The two ground-truth
# baselines get distinct labels so tables that show both stay unambiguous.
MODEL_LABEL = {
    "omni_fish-s2-pro": "fish-s2-pro",
    "omni_qwen3-tts-1.7b": "qwen3-1.7b",
    "omni_qwen3-tts-0.6b": "qwen3-0.6b",
    "omni_voxcpm2": "voxcpm2",
    "omni_voxtral-tts": "voxtral",
    "omni_moss-tts-nano": "moss-nano",
    "elevenlabs_v4_clone": "elevenlabs-v4",
    "cartesia_sonic-3.6_clone": "cartesia-sonic-3.6",
    "inworld_2_clone": "inworld-tts-2",
    "gemini_3.8-flash": "gemini-3.8-flash",
    "gt_clean": "gt (clean)",
    "gt_in-the-wild": "gt (in-the-wild)",
    "whisper_clean": "whisper (clean)",
    "whisper_in-the-wild": "whisper (in-the-wild)",
}


def label(model_dir):
    """Display label for a results/ model directory name."""
    if model_dir in MODEL_LABEL:
        return MODEL_LABEL[model_dir]
    for pre in ("omni_", "gt_"):
        if model_dir.startswith(pre):
            return model_dir[len(pre):]
    return model_dir


# --- conditioning classes ---------------------------------------------------
# What each system is actually conditioned on. This is HARNESS POLICY, not a
# model property alone, and it differs per system — derived from the generation
# code, not from model cards:
#   - scripts/omni_models.sh reftext_policy(): which vLLM-Omni clone models get
#     `--no-ref-text` (audio-only) vs keep the reference transcript;
#   - scripts/generate_omni.py: `--x-vector-only` sets x_vector_only_mode=True
#     (the qwen3-tts Base task requires it to accept --no-ref-text);
#   - scripts/generate_zonos2.py: audio-only clone via speaker_audio_base64,
#     fixed --seed 0, vendor sampling temperature 1.15;
#   - the native generate_*.py recipes (kokoro/neutts/vibevoice/moss/cosyvoice).
#
# Classes:
#   with-transcript : clone conditioned on ref wav + its transcript
#   no-transcript   : clone conditioned on ref wav audio only
#   x-vector        : clone via speaker embedding only (x_vector_only_mode)
#   preset          : fixed built-in voice — NOT cloning the prompt speaker
#
# Seeding note: zonos2_* and zonos3-mini_* run with a fixed seed (0); the vLLM-Omni-served
# systems pass no seed/temperature (generate_omni.py defaults are None), i.e.
# unseeded server-default sampling; native recipes note their own defaults. Hosted APIs:
# elevenlabs_* send seed 0 (provider best-effort); the other hosted APIs expose no seed.
CONDITIONS = {
    # zonos2 server (scripts/generate_zonos2.py): audio-only clone.
    "zonos2_accurate": ("no-transcript",
                        "audio-only clone; accurate_mode=true; seed 0, temp 1.15 vendor sampling"),
    "zonos2_expressive": ("no-transcript",
                          "audio-only clone; accurate_mode=false; seed 0, temp 1.15 vendor sampling"),
    # ZONOS3 Mini via the release minisgl server (scripts/generate_zonos3.py -> /tts/generate): audio-only clone.
    "zonos3-mini_reference": ("no-transcript",
                              "release server, raw reference prefix (speaker_mode=reference), no transcript; seed 0, "
                              "T 0.75 top-k 80 top-p 0.69, rep 96×2.4, CFG 1, 0.2 s silence prefill, "
                              "server TN for en zh de es fr it pt ja ko"),
    "zonos3-mini_embedding": ("x-vector",
                              "release server, ownmect_s2 speaker embedding only (speaker_mode=embedding), "
                              "no transcript; sampling as zonos3-mini_reference"),
    # As above plus LACI (laci=true; alignment-guarded decoding with rollbacks), served from the z3m-laci
    # runtime tree (d24d592).
    "zonos3-mini-laci_reference": ("no-transcript",
                                   "as zonos3-mini_reference plus LACI (laci=true; ZTTS3-Eval assets/laci heads: "
                                   "zh heads for cmn, en heads for every other server language)"),
    "zonos3-mini-laci_embedding": ("x-vector",
                                   "as zonos3-mini_embedding plus LACI (laci=true; heads as zonos3-mini-laci_reference)"),
    # Internal ZONOS3 Mini checkpoints, same client and sampling as zonos3-mini_*, served from gabriel's minisgl
    # (inference-frozen-arch-20261012 @ 65ccd72 + per-table condition bucket counts [10, 10, 30]).
    "zonos3-rate30-20k_reference": ("no-transcript",
                                    "internal checkpoint: omni3b 60k recipe + 20k steps with 30 speaking-rate "
                                    "buckets; as zonos3-mini_reference otherwise"),
    "zonos3-rate30-20k_embedding": ("x-vector",
                                    "internal checkpoint as zonos3-rate30-20k_reference; as zonos3-mini_embedding "
                                    "otherwise"),
    "zonos3-rate30-20k-r2grpo_reference": ("no-transcript",
                                           "internal checkpoint: zonos3-rate30-20k + tail DPO + round-2 GRPO "
                                           "(WER-first reward, step 587); as zonos3-mini_reference otherwise"),
    "zonos3-rate30-20k-r2grpo_embedding": ("x-vector",
                                           "internal checkpoint as zonos3-rate30-20k-r2grpo_reference; as "
                                           "zonos3-mini_embedding otherwise"),
    # Continuation in-process (no server: /tts/generate cannot prefill the prompt's frames), gabriel's minisgl @ 2c78070.
    "zonos3-rate30-20k-r2grpo-reftext_embedding": ("with-transcript",
                                                   "in-process TTSLLM (scripts/generate_zonos3_reftext.py): prompt "
                                                   "transcript + target text, the prompt's codec frames (edge "
                                                   "silence trimmed, top_db 35; last 2 frames, the clip's end-of-file "
                                                   "edge, dropped) prefilled after the 0.2 s silence "
                                                   "prefill (output = audio after the prefill), plus the ownmect_s2 "
                                                   "speaker embedding of the untrimmed prompt; checkpoint as "
                                                   "zonos3-rate30-20k-r2grpo_reference; sampling as "
                                                   "zonos3-mini_reference"),
    # vLLM-Omni served (scripts/omni_models.sh reftext_policy + generate_omni.py);
    # all unseeded, server-default sampling.
    "omni_voxcpm2": ("no-transcript", "--no-ref-text (reftext_policy); unseeded server defaults"),
    "omni_omnivoice": ("no-transcript", "--no-ref-text (reftext_policy); unseeded server defaults"),
    "omni_qwen3-tts-0.6b": ("x-vector",
                            "--no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults"),
    "omni_qwen3-tts-1.7b": ("x-vector",
                            "--no-ref-text --x-vector-only → x_vector_only_mode=True; unseeded server defaults"),
    "omni_omnivoice-reftext": ("with-transcript",
                               "same weights as omnivoice, ref_text kept (separate key); unseeded server defaults"),
    "omni_qwen3-tts-0.6b-reftext": ("with-transcript",
                                    "same weights as qwen3-tts-0.6b, ref_text kept (separate key); unseeded server defaults"),
    "omni_qwen3-tts-1.7b-reftext": ("with-transcript",
                                    "same weights as qwen3-tts-1.7b, ref_text kept (separate key); unseeded server defaults"),
    "omni_voxcpm2-reftext": ("with-transcript",
                             "same weights as voxcpm2, ref_text kept (separate key); unseeded server defaults"),
    "omni_fish-s2-pro": ("with-transcript",
                         "ref_text kept (server rejects no-ref-text); unseeded server defaults"),
    "omni_glm-tts": ("with-transcript",
                     "ref_text kept (default policy, no-ref-text unverified); unseeded server defaults"),
    "omni_higgs-audio-v3": ("with-transcript",
                            "ref_text kept (upstream prescribes ref_audio + ref_text); unseeded server defaults"),
    "omni_drama-3": ("with-transcript",
                     "ref_text kept (server requires ref_audio + ref_text); vllm-omni 0.32.0 on vllm "
                     "0.32.0; unseeded server defaults (T 1.0 top-k 30 top-p 0.9)"),
    "omni_breeze-tts-2": ("with-transcript",
                          "ref_text kept (server requires it); vllm-omni 0.32.0 breeze_tts_2_throughput "
                          "profile (32 concurrent); server sampling T 0.9 top-k 50 rep 1.1, CFG 1, seed 42 "
                          "per request; en zh only"),
    "omni_voxtral-tts": ("preset", "preset mode (task_type=CustomVoice); unseeded server defaults"),
    # Upstream SGLang-Omni adapter server (scripts/generate_audio8.py -> /v1/audio/speech).
    "audio8-tts-realistic-0.6b": ("with-transcript",
                                  "SGLang-Omni adapter (Audio8_TTS@4eb54a6, sglang-omni@68a5723); prompt wav + "
                                  "transcript; T 0.8 top-p 0.95 top-k 50, ≤1024 frames; unseeded server sampling"),
    # Native in-process recipes.
    "cosyvoice3": ("with-transcript",
                   "native upstream; prompt wav + transcript (endofprompt-terminated system prompt; ja→katakana)"),
    "ming-3.0-tts-0.5b": ("with-transcript",
                          "native upstream (Ming-3.0-tts@a39b806, generate_ming3.py); prompt wav + transcript + "
                          "CAM++ speaker embedding, raw text; greedy text decoding, CFG 2.0, sigma 0.25, seed "
                          "1895 per utt; zh en de es fr ja ko only"),
    "ming-3.0-tts-tiny": ("with-transcript",
                          "native upstream (Ming-3.0-tts@a39b806, generate_ming3.py; seed / multilingual "
                          "tasks); prompt wav + transcript, raw text; formal settings CFG 2.0, temp 0, 10 flow "
                          "steps, sigma 0.25, seed 1234 per utt; zh en de es fr ja ko only"),
    # Same runs on prompts with edge silence trimmed (librosa top_db 35, generate_ming3.py
    # --trim-prompt): Ming continues from the prompt's tail and mostly emits ~0.3 s of
    # silence on the FLEURS-R prompts' trailing silence. Harness change for Ming only.
    "ming-3.0-tts-0.5b-trim": ("with-transcript",
                               "as ming-3.0-tts-0.5b, prompt wav edge silence trimmed (top_db 35) before "
                               "encoding"),
    "ming-3.0-tts-tiny-trim": ("with-transcript",
                               "as ming-3.0-tts-tiny, prompt wav edge silence trimmed (top_db 35) before "
                               "encoding"),
    "moss-tts": ("no-transcript",
                 "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-v1.5": ("no-transcript",
                      "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-local": ("no-transcript",
                       "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "moss-tts-local-v1.5": ("no-transcript",
                            "audio-only clone (generate_moss_tts.py, no transcript); temp 0.9 reference-app sampling"),
    "vibevoice-1.5b": ("no-transcript",
                       "voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps"),
    "vibevoice-7b": ("no-transcript",
                     "voice-prompt clone (voice_samples=[ref wav], no transcript); cfg 1.3, 10 DDPM steps"),
    "pocket-tts": ("no-transcript",
                   "official pocket-tts 4.0.1 on CPU, kyutai/pocket-tts@db04f0e6 default model per language "
                   "(6-layer); audio-only clone of the prompt wav; temp 0.3, EOS -4, torch seed 42 per utt; "
                   "en fr de pt it es only"),
    "kokoro": ("preset", "built-in voice per language; prompt wav ignored"),
    "phonon-40m-multilingual": ("preset",
                                "ptts 0.6.0 Rust runtime, q8 GGUF, CPU; gradium/phonon-voices catalog voice of "
                                "the utterance's language, deterministic per-utt rotation (no local cloning); "
                                "runtime text normalization; temp 0.3, default seed; en fr de es pt only"),
    "neutts-2e": ("preset", "4 fixed speakers, deterministic per-utt rotation; prompt wav ignored"),
    "vibevoice-rt-0.5b": ("preset", "preset speaker embeddings, per-utt rotation; prompt wav ignored"),
    # Hosted APIs (scripts/generate_{elevenlabs_clone,cartesia,inworld,gemini}.py)
    "elevenlabs_v4_clone": ("no-transcript",
                            "hosted API eleven_v4 via /v1/text-to-dialogue; per-utterance instant voice clone "
                            "of the prompt wav (no transcript, no denoise), deleted after use; language_code "
                            "sent (nb→no, tl→fil); seed 0 (provider best-effort); wav_24000"),
    "cartesia_sonic-3.6_clone": ("no-transcript",
                                 "hosted API sonic-3.6-2026-08-27 (/tts/bytes, Cartesia-Version 2026-08-14); "
                                 "per-utterance instant clone of the prompt wav (no transcript), deleted after "
                                 "use; language sent where supported (nb→no, fil→tl; ca/fa unsupported: clone "
                                 "language en, none sent); provider-default sampling"),
    "inworld_2_clone": ("no-transcript",
                        "hosted API inworld-tts-2 (/tts/v1/voice); per-utterance instant voice clone of the "
                        "prompt wav (no transcription; TTS-2 cloning ignores it), deleted after use; language "
                        "+ clone languageCode sent (tl→fil); deliveryMode default (BALANCED); LINEAR16"),
    "gemini_3.8-flash": ("preset",
                         "hosted API gemini-3.8-flash-tts (generateContent); 30 prebuilt voices, deterministic "
                         "per-utt rotation; prompt wav ignored (voice replication requires a consent "
                         "recording from the same speaker); language auto-detected"),
}

# Result-table markers per class (with-transcript is the unmarked default).
COND_MARKER = {"no-transcript": "†", "x-vector": "◇", "preset": "‡"}
COND_LEGEND = ("† = no-transcript clone (audio-only) · ◇ = x-vector-only clone · "
               "‡ = preset voice (not cloning the prompt) · unmarked = ref transcript given. "
               "See §1b.")


def condition(model_dir):
    """(class, provenance) for a results/ model dir, or None (gt_*/whisper_*).

    The moss-tts prefix fallback covers moss-tts variants without an explicit entry.
    """
    if model_dir in CONDITIONS:
        return CONDITIONS[model_dir]
    if model_dir.startswith(("gt_", "whisper_")):
        return None
    if model_dir.startswith("moss-tts"):
        return ("no-transcript", "audio-only clone (generate_moss_tts.py, no transcript)")
    return None


def marked_label(model_dir):
    """Display label plus the conditioning-class marker (†/◇/‡) — use this in
    every ranking table so no-transcript / x-vector / preset systems are never
    read against with-transcript ones undisclosed."""
    cond = condition(model_dir)
    if cond is None:
        return label(model_dir)
    return label(model_dir) + COND_MARKER.get(cond[0], "")


# --- release dates ------------------------------------------------------------
# Release of the weights each results/ dir was generated from: (YYYY-MM-DD, source).
# Used to plot the capability index over time. Dates come from
# the project's own announcement or dated news log, not Hugging Face `createdAt`: repos
# are often created privately weeks before launch (Voxtral-4B-TTS-2603: created
# 2025-11-17, released 2026-03-23). Variants of one checkpoint (-reftext, _embedding,
# -laci, accurate/expressive) share its date. A system without an entry is undated
# and left out of the time plot. Hosted API systems (no released weights) are dated by
# the vendor's public API launch.
RELEASES = {
    "kokoro": ("2025-01-27", "hexgrad/Kokoro-82M model card, Releases table (v1.0)"),
    "vibevoice-1.5b": ("2025-08-25", "microsoft/VibeVoice README news (VibeVoice-TTS open-sourced)"),
    "vibevoice-7b": ("2025-08-25", "microsoft/VibeVoice README news (VibeVoice-TTS; Large = 7B)"),
    "vibevoice-rt-0.5b": ("2025-12-03", "microsoft/VibeVoice README news (VibeVoice-Realtime-0.5B)"),
    "omni_glm-tts": ("2025-12-11", "zai-org/GLM-TTS README news (open-sourced)"),
    "cosyvoice3": ("2025-12-11", "HF FunAudioLLM/Fun-CosyVoice3-0.5B-2512 first commit (README undated)"),
    "omni_qwen3-tts-0.6b": ("2026-01-22", "QwenLM/Qwen3-TTS README news"),
    "omni_qwen3-tts-0.6b-reftext": ("2026-01-22", "QwenLM/Qwen3-TTS README news"),
    "omni_qwen3-tts-1.7b": ("2026-01-22", "QwenLM/Qwen3-TTS README news"),
    "omni_qwen3-tts-1.7b-reftext": ("2026-01-22", "QwenLM/Qwen3-TTS README news"),
    "moss-tts": ("2026-02-10", "OpenMOSS/MOSS-TTS README news (MOSS-TTS Family)"),
    "moss-tts-local": ("2026-02-10", "OpenMOSS/MOSS-TTS README news (MOSS-TTS Family)"),
    "omni_fish-s2-pro": ("2026-03-09", "Fish Audio blog, 'Fish Audio Open-Sources S2'"),
    "omni_voxtral-tts": ("2026-03-23", "Mistral blog, 'Speaking of Voxtral'"),
    "omni_omnivoice": ("2026-04-01", "arXiv 2604.00688 v1 (OmniVoice)"),
    "omni_omnivoice-reftext": ("2026-04-01", "arXiv 2604.00688 v1 (OmniVoice)"),
    "omni_voxcpm2": ("2026-04-03", "HF openbmb/VoxCPM2 first commit (README news: 2026.04)"),
    "omni_voxcpm2-reftext": ("2026-04-03", "HF openbmb/VoxCPM2 first commit (README news: 2026.04)"),
    "inworld_2_clone": ("2026-05-05", "Inworld TTS release notes, 'Realtime TTS-2'"),
    "moss-tts-v1.5": ("2026-05-26", "OpenMOSS/MOSS-TTS README news"),
    "omni_higgs-audio-v3": ("2026-06-04", "Boson AI blog, 'Higgs Audio v3 TTS'"),
    "zonos2_accurate": ("2026-06-12", "Zyphra blog, 'ZONOS2'"),
    "zonos2_expressive": ("2026-06-12", "Zyphra blog, 'ZONOS2'"),
    "moss-tts-local-v1.5": ("2026-06-18", "OpenMOSS/MOSS-TTS README news"),
    "neutts-2e": ("2026-07-30", "Neuphonic blog, 'Introducing NeuTTS-2E'"),
    "omni_breeze-tts-2": ("2026-08-25", "BreezeBlue/Breeze-TTS-2 card news (weights + PyTorch inference code)"),
    "cartesia_sonic-3.6_clone": ("2026-08-27", "Cartesia docs, Sonic 3.6 dated snapshot sonic-3.6-2026-08-27 (Stable)"),
    "gemini_3.8-flash": ("2026-09-23", "Google blog, 'Gemini 3.8 text-to-speech says hello'"),
    "elevenlabs_v4_clone": ("2026-09-28", "ElevenLabs blog, 'Eleven v4'"),
    "pocket-tts": ("2026-10-11", "HF kyutai/pocket-tts commit c54c3637 (default english refreshed; other "
                   "languages 2026-10-01)"),
    "omni_drama-3": ("2026-10-13", "Fish Audio blog, 'Fish Audio Drama 3' (HF fishaudio/drama-3 public 2026-10-13)"),
    "ming-3.0-tts-0.5b": ("2026-10-19", "inclusionAI/Ming-3.0-tts GitHub, 'Initial release of Ming-3.0-tts'"),
    "ming-3.0-tts-tiny": ("2026-10-19", "inclusionAI/Ming-3.0-tts GitHub, 'Initial release of Ming-3.0-tts'"),
    "ming-3.0-tts-0.5b-trim": ("2026-10-19", "inclusionAI/Ming-3.0-tts GitHub, 'Initial release of Ming-3.0-tts'"),
    "ming-3.0-tts-tiny-trim": ("2026-10-19", "inclusionAI/Ming-3.0-tts GitHub, 'Initial release of Ming-3.0-tts'"),
    "phonon-40m-multilingual": ("2026-10-20", "Gradium blog, 'Open-Sourcing Phonon'"),
    "audio8-tts-realistic-0.6b": ("2026-10-29", "Edge0 launch post on X, 'Introducing Audio8-TTS-Realistic' (HF repo created 2026-10-14)"),
    "zonos3-mini_reference": ("2026-11-06", "Zyphra release plan"),
    "zonos3-mini_embedding": ("2026-11-06", "Zyphra release plan"),
    "zonos3-mini-laci_reference": ("2026-11-06", "Zyphra release plan"),
    "zonos3-mini-laci_embedding": ("2026-11-06", "Zyphra release plan"),
    "zonos3-rate30-20k_reference": ("2026-11-06", "Zyphra release plan"),
    "zonos3-rate30-20k_embedding": ("2026-11-06", "Zyphra release plan"),
    "zonos3-rate30-20k-r2grpo_reference": ("2026-11-06", "Zyphra release plan"),
    "zonos3-rate30-20k-r2grpo_embedding": ("2026-11-06", "Zyphra release plan"),
    "zonos3-rate30-20k-r2grpo-reftext_embedding": ("2026-11-06", "Zyphra release plan"),
}

# --- parameter counts -----------------------------------------------------------
# Published size of each system's main generative model (LM / backbone), in billions:
# (total, active or None, source). Codecs and vocoders are excluded unless the
# developer's own headline number includes them, so this is the size a model is known
# by, not every weight it loads (Hugging Face repo totals mix both: VibeVoice-1.5B's
# repo holds 2.7B with its tokenizers). For mixture-of-experts models `active` is the
# per-token count; the chart plots `total`. Variants of one checkpoint share an entry.
# Hosted API systems publish no size and have no entry.
_Z3M = (0.886, 0.155, "release checkpoint lm/mect_60k tensor count (64 experts, top-2 routed)")
# rate30 and its post-trains: the lm/mect_60k architecture plus 20 more speaking-rate bucket rows (0.88617 B).
_Z3M_RATE30 = (0.886, 0.155, "zonos3-rate30-20k-r2grpo converted model.pth tensor count (lm/mect_60k arch, 30 rate buckets)")
PARAMS = {
    "kokoro": (0.082, None, "model name (Kokoro-82M)"),
    "phonon-40m-multilingual": (0.04, None, "gradium/phonon-40m-multilingual card: 40M excluding the audio tokenizer"),
    "pocket-tts": (0.1, None, "kyutai/pocket-tts card: '100M parameters'"),
    "neutts-2e": (0.125, None, "Neuphonic blog, '125M-parameter'"),
    "cosyvoice3": (0.5, None, "model name (Fun-CosyVoice3-0.5B)"),
    "vibevoice-rt-0.5b": (0.5, None, "model name (VibeVoice-Realtime-0.5B)"),
    "vibevoice-1.5b": (1.5, None, "model name (VibeVoice-1.5B)"),
    "vibevoice-7b": (7.0, None, "model name (VibeVoice-7B / Large)"),
    "omni_glm-tts": (1.55, None, "zai-org/GLM-TTS llm/ weights: 6.2 GB fp32 Llama (28 L, d 2048)"),
    "omni_qwen3-tts-0.6b": (0.6, None, "model name (Qwen3-TTS-12Hz-0.6B)"),
    "omni_qwen3-tts-0.6b-reftext": (0.6, None, "model name (Qwen3-TTS-12Hz-0.6B)"),
    "omni_qwen3-tts-1.7b": (1.7, None, "model name (Qwen3-TTS-12Hz-1.7B)"),
    "omni_qwen3-tts-1.7b-reftext": (1.7, None, "model name (Qwen3-TTS-12Hz-1.7B)"),
    "moss-tts": (8.0, None, "OpenMOSS/MOSS-TTS README (MossTTSDelay 8B)"),
    "moss-tts-v1.5": (8.0, None, "OpenMOSS/MOSS-TTS README (MossTTSDelay 8B)"),
    "moss-tts-local": (1.7, None, "OpenMOSS/MOSS-TTS README (MossTTSLocal 1.7B)"),
    "moss-tts-local-v1.5": (4.0, None, "OpenMOSS/MOSS-TTS README news (4B MossTTSLocal)"),
    "omni_fish-s2-pro": (4.4, None, "fishaudio/s2-pro card: slow AR 4B + fast AR 0.4B"),
    "omni_voxtral-tts": (4.0, None, "model name (Voxtral-4B-TTS-2603)"),
    "omni_omnivoice": (0.6, None, "k2-fsa/OmniVoice card: Qwen3-0.6B backbone"),
    "omni_omnivoice-reftext": (0.6, None, "k2-fsa/OmniVoice card: Qwen3-0.6B backbone"),
    "omni_voxcpm2": (2.0, None, "openbmb/VoxCPM2 card: '2B parameters'"),
    "omni_voxcpm2-reftext": (2.0, None, "openbmb/VoxCPM2 card: '2B parameters'"),
    "omni_higgs-audio-v3": (4.0, None, "model name (higgs-audio-v3-tts-4b)"),
    "omni_drama-3": (35.0, 3.0, "fishaudio/drama-3 card: 35B MoE, 3B active per token"),
    "omni_breeze-tts-2": (3.48, None, "BreezeBlue/Breeze-TTS-2 checkpoint: 6.97 GB bf16 (Qwen3 backbone, T5Gemma2 text encoder, depth decoder); no published size"),
    "ming-3.0-tts-0.5b": (0.5, None, "inclusionAI/Ming-3.0-tts README: 0.5b dense"),
    "ming-3.0-tts-tiny": (7.9, 1.3, "inclusionAI/Ming-3.0-tts README: tiny MoE, 7.9B total, 1.3B active"),
    "ming-3.0-tts-0.5b-trim": (0.5, None, "inclusionAI/Ming-3.0-tts README: 0.5b dense"),
    "ming-3.0-tts-tiny-trim": (7.9, 1.3, "inclusionAI/Ming-3.0-tts README: tiny MoE, 7.9B total, 1.3B active"),
    "audio8-tts-realistic-0.6b": (0.601, None, "Edge0/Audio8-TTS-Realistic-0.6b card: 601,159,424 main-model params, codec excluded"),
    "zonos2_accurate": (8.0, 0.9, "Zyphra blog, 'ZONOS2'"),
    "zonos2_expressive": (8.0, 0.9, "Zyphra blog, 'ZONOS2'"),
    "zonos3-mini_reference": _Z3M,
    "zonos3-mini_embedding": _Z3M,
    "zonos3-mini-laci_reference": _Z3M,
    "zonos3-mini-laci_embedding": _Z3M,
    "zonos3-rate30-20k_reference": _Z3M_RATE30,
    "zonos3-rate30-20k_embedding": _Z3M_RATE30,
    "zonos3-rate30-20k-r2grpo_reference": _Z3M_RATE30,
    "zonos3-rate30-20k-r2grpo_embedding": _Z3M_RATE30,
    "zonos3-rate30-20k-r2grpo-reftext_embedding": _Z3M_RATE30,
}

# --- system discovery -------------------------------------------------------
def discover_systems(task, root=RESULTS_ROOT):
    """{label: results-dir} for every model with a `<root>/<model>/<task>/` leaf.

    First directory wins on a label collision (sorted order).
    """
    out = {}
    if not os.path.isdir(root):
        return out
    for mdir in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, mdir, task)):
            continue
        out.setdefault(label(mdir), mdir)
    return out


# --- language baskets -------------------------------------------------------
# Display order: the ITW basket first (it contains the core-9), then the Clean-only
# languages in basket order. Both baskets are frozen in utils/benchmark.py. Languages
# outside both sort alphabetically after these.
CANON_LANG_ORDER = list(ITW_LANGS) + [l for l in CLEAN_LANGS if l not in ITW_LANGS]


def ordered_langs(langs):
    """Sort a language collection into the canonical display order."""
    order = {l: i for i, l in enumerate(CANON_LANG_ORDER)}
    return sorted(langs, key=lambda l: (order.get(l, len(order)), l))


def data_langs(root_data):
    """Languages present under a data/<task>/ dir (those with a `text` file),
    in canonical display order — the language basket auto-grows with the data."""
    if not os.path.isdir(root_data):
        return []
    return ordered_langs(
        d for d in os.listdir(root_data)
        if os.path.isfile(os.path.join(root_data, d, "text")))

# --- snapshot access + the macro rule ---------------------------------------
# One loader for the snapshot build_results.py writes, shared by gen_md.py and
# plot_tradeoff.py so the published table and the published figure can never
# disagree on what a "macro" is.
SNAPSHOT_PATH = os.path.join(SCRIPT_DIR, "_cache", "results_snapshot.json")


def load_snapshot(path=SNAPSHOT_PATH):
    """The record list written by build_results.py."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_records(recs):
    """{(model, task, set): record}. gt_* baselines carry task == "" (flat dirs)."""
    return {(r["model"], r["task"], r["set"]): r for r in recs}


def rec_value(rec, field):
    """Scalar of a snapshot field, or None: WER_CLIP for `wer`; NaN counts as missing."""
    if rec is None:
        return None
    v = rec.get(field)
    if field == "wer":
        v = v[2] if v else None
    if v is None or v != v:
        return None
    return v


def lang_values(idx, model, task, field, langs):
    """Per-language values of `field` for a system, None where missing."""
    t = "" if model.startswith("gt_") else task
    return [rec_value(idx.get((model, t, l)), field) for l in langs]


def macro(idx, model, task, field, langs):
    """Unweighted mean over `langs` (the task's frozen basket, utils.benchmark.langs_for),
    defined ONLY when every basket language has the value: a mean over a different
    language subset is a different quantity and must not share a column with the full one."""
    vals = lang_values(idx, model, task, field, langs)
    if not vals or any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def n_covered(idx, model, task, field, langs):
    """How many basket languages have a value for `field`."""
    return sum(v is not None for v in lang_values(idx, model, task, field, langs))



# --- atomic JSON ------------------------------------------------------------
def write_json_atomic(path, obj, **json_kw):
    """json.dump to a temp file in the same directory, then os.replace().

    Readers (the explorer, plot scripts) can never observe a half-written
    file, and a crash leaves the previous version intact.
    """
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **json_kw)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
