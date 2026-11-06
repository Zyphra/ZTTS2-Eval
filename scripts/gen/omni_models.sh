# scripts/gen/omni_models.sh -- shared vLLM-Omni TTS model registry.
#
# SOURCE this (`. scripts/gen/omni_models.sh`); do not execute it. scripts/gen/run_omni.sh
# sources this single copy for every task.

# vLLM-Omni TTS models. Each spec is "<hf-model-id>|<mode>|<deploy-config yaml>".
#   mode clone  -> zero-shot voice cloning (task_type=Base, reference wav + transcript)
#   mode preset -> built-in voices (task_type=CustomVoice); design -> VoiceDesign
# The deploy-config YAML names ship in vllm_omni/deploy/. HF ids are BEST-EFFORT;
# smoke-test one model with `--limit 3` first (a wrong id/yaml fails at startup).
ALL_MODELS="qwen3-tts-1.7b qwen3-tts-0.6b cosyvoice3 voxcpm2 voxtral-tts fish-s2-pro ming-flash-omni-tts omnivoice"

# Newer TTS models shipped by a LATER vllm-omni. Wired in model_spec() and runnable by
# explicit --model, but kept OUT of ALL_MODELS (the `--model all` sweep) until
# validated. To enable: bump the runtime, smoke one, then promote.
NEW_MODELS="glm-tts ming-tts moss-tts higgs-audio-v3 drama-3 breeze-tts-2"

# RETIRED -- kept in model_spec() so an explicit --model still works, but out of both
# sweep lists so `--model all` can never resurrect them, and their results/ trees were
# deleted:
#   moss-tts-nano  decode bug -- 3 usable utterances out of 1279 on clean/en
#   indextts2      its diffusion DiT flash-attn kernel PTX-mismatches the driver; the
#                  server reaches READY, then the first request kills the engine -> 0 wavs

model_spec() {  # key -> "<hf-model-id>|<mode>|<deploy-config yaml>"
    case "$1" in
        qwen3-tts-1.7b)      echo "Qwen/Qwen3-TTS-12Hz-1.7B-Base|clone|qwen3_tts.yaml" ;;
        qwen3-tts-0.6b)      echo "Qwen/Qwen3-TTS-12Hz-0.6B-Base|clone|qwen3_tts.yaml" ;;
        cosyvoice3)          echo "FunAudioLLM/Fun-CosyVoice3-0.5B-2512|clone|cosyvoice3.yaml" ;;
        voxcpm2)             echo "openbmb/VoxCPM2|clone|voxcpm2.yaml" ;;
        voxtral-tts)         echo "mistralai/Voxtral-4B-TTS-2603|preset|voxtral_tts.yaml" ;;
        moss-tts-nano)       echo "OpenMOSS-Team/MOSS-TTS-Nano|clone|moss_tts_nano.yaml" ;;
        fish-s2-pro)         echo "fishaudio/s2-pro|clone|fish_qwen3_omni.yaml" ;;
        ming-flash-omni-tts) echo "inclusionAI/Ming-flash-omni-Preview|clone|ming_flash_omni_tts.yaml" ;;
        omnivoice)           echo "k2-fsa/OmniVoice|clone|omnivoice.yaml" ;;
        # omnivoice-reftext: same model, WITH ref_text (comparison vs no-ref-text).
        # Separate key -> its own results/omni_omnivoice-reftext/ dir.
        omnivoice-reftext)   echo "k2-fsa/OmniVoice|clone|omnivoice.yaml" ;;
        # 2026-08 methodology review: symmetric with-transcript rows for the other
        # no-ref-text systems, so omnivoice-reftext is no longer the only rescued
        # variant in the rankings. Same idiom: same model, reftext_policy's default
        # case keeps ref_text (and for qwen3, no forced x_vector_only_mode).
        qwen3-tts-1.7b-reftext) echo "Qwen/Qwen3-TTS-12Hz-1.7B-Base|clone|qwen3_tts.yaml" ;;
        qwen3-tts-0.6b-reftext) echo "Qwen/Qwen3-TTS-12Hz-0.6B-Base|clone|qwen3_tts.yaml" ;;
        voxcpm2-reftext)        echo "openbmb/VoxCPM2|clone|voxcpm2.yaml" ;;
        # --- newer TTS models: require the vllm-omni upgrade (NEW_MODELS) ---
        indextts2)           echo "IndexTeam/IndexTTS-2|clone|indextts2.yaml" ;;
        glm-tts)             echo "zai-org/GLM-TTS|clone|glm_tts.yaml" ;;
        ming-tts)            echo "inclusionAI/Ming-omni-tts-0.5B|clone|ming_tts.yaml" ;;
        moss-tts)            echo "OpenMOSS-Team/MOSS-TTS|clone|moss_tts.yaml" ;;
        # Higgs-Audio V3 (Boson AI). Has NO built-in speakers -> it's a zero-shot cloner
        # (server: "use ref_audio + ref_text"), so clone mode with ref_text kept. Needs
        # --trust-remote-code (extra_serve_args), DEEP_GEMM disabled (extra_env), and
        # base64 ref-audio (refaudio_mode; qwen3 multimodal connector rejects file://).
        higgs-audio-v3)      echo "bosonai/higgs-audio-v3-tts-4b|clone|higgs_multimodal_qwen3_high_throughput.yaml" ;;
        # Fish Audio Drama 3 (35B MoE, 3B active). Needs vllm-omni >= 0.32 on vllm >= 0.31
        # (model_type fish_omni), newer than the repo .venv: run it with OMNI_VENV pointing
        # at a separate venv. The benchmark run used (CUDA 12.8 driver -> cu129 builds;
        # PyPI's torchcodec 0.17 is a CUDA 13 build that fails to load on such nodes):
        #   uv venv --python 3.12 ~/.venvs/vllm-omni-drama3
        #   VIRTUAL_ENV=~/.venvs/vllm-omni-drama3 uv pip install --index-strategy \
        #     unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
        #     torch==2.13.0+cu129 torchaudio==2.11.0+cu129 torchvision==0.28.0+cu129 \
        #     torchcodec==0.16.0+cu129 vllm-omni==0.32.0 \
        #     https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
        # Cloning requires ref_audio + ref_text, so reftext_policy's default (ref_text
        # kept) applies.
        drama-3)             echo "fishaudio/drama-3|clone|fish_omni.yaml" ;;
        # Breeze TTS 2 (BreezeBlue). vllm-omni >= 0.32 (same venv as drama-3) serves it with
        # continuous batching: the non-streaming throughput profile batches 32 requests per
        # GPU with CUDA graphs, where the upstream PyTorch runtime (github.com/breezeblue-ai/
        # breeze-tts) clones one request at a time (its batch API shares one reference across
        # all texts). Sampling is the checkpoint's: T 0.9, top-k 50, repetition penalty 1.1,
        # seed 42, CFG 1. Cloning requires the reference transcript; English and Chinese only.
        breeze-tts-2)        echo "BreezeBlue/Breeze-TTS-2|clone|breeze_tts_2_throughput.yaml" ;;
        *)                   echo "" ;;
    esac
}

# Per-model reference-text policy for clone mode. We OMIT ref_text (audio-only
# cloning) wherever the server supports it. Returns extra generate_omni.py args:
#   ""                              -> keep ref_text (model REQUIRES it)
#   "--no-ref-text"                 -> audio-only cloning
#   "--no-ref-text --x-vector-only" -> audio-only, needs x_vector_only_mode
# qwen3-tts-* need --x-vector-only; voxcpm2/moss-tts-nano/omnivoice take plain
# --no-ref-text; cosyvoice3/fish-s2-pro REJECT no-ref-text server-side. New clone
# models default to ref_text until a no-ref-text smoke proves otherwise.
# Env overrides: OMNI_FORCE_REFTEXT=1 forces ref_text; OMNI_FORCE_NOREFTEXT=1 forces
# plain --no-ref-text, for every model.
reftext_policy() {  # key -> extra generate_omni.py args
    [ "${OMNI_FORCE_REFTEXT:-0}" = 1 ]   && { echo ""; return; }
    [ "${OMNI_FORCE_NOREFTEXT:-0}" = 1 ] && { echo "--no-ref-text"; return; }
    case "$1" in
        qwen3-tts-1.7b|qwen3-tts-0.6b)   echo "--no-ref-text --x-vector-only" ;;
        voxcpm2|moss-tts-nano|omnivoice) echo "--no-ref-text" ;;
        cosyvoice3|fish-s2-pro)          echo "" ;;   # reject no-ref-text server-side
        *)                               echo "" ;;   # unverified -> keep ref_text
    esac
}

# How the clone reference wav is delivered: 'url' = a file:// path the server opens
# locally (needs --allowed-local-media-path); 'base64' = inline data: URI. OmniVoice
# and IndexTTS-2 route ref_audio through vLLM's generic multimodal connector, which
# rejects file://, so they need base64.
refaudio_mode() {  # key -> url | base64
    case "$1" in
        omnivoice|omnivoice-reftext) echo "base64" ;;
        indextts2)                   echo "base64" ;;
        higgs-audio-v3)              echo "base64" ;;   # qwen3 multimodal connector
        *)                           echo "url" ;;
    esac
}

# Extra source trees a model's server needs on PYTHONPATH (code that isn't a pip
# package). fish-s2-pro's DAC codec (fish_speech.models.dac.modded_dac) lives in the
# fish-speech checkout (github.com/fishaudio/fish-speech), not the venv; without it the server dies at READY with
# "ModuleNotFoundError: No module named 'fish_speech'". Set FISH_SPEECH_TREE to it.
extra_pythonpath() {  # key -> path to prepend to PYTHONPATH (or empty)
    case "$1" in
        fish-s2-pro) echo "${FISH_SPEECH_TREE:-}" ;;
        *)           echo "" ;;
    esac
}

# Extra `vllm-omni serve` flags a model needs (from the vllm-omni recipes/). GLM-TTS,
# IndexTTS-2 and VoxCPM2 require --trust-remote-code: the model weights load fine, but
# engine-core init builds the guided-decoding tokenizer via AutoTokenizer, which
# re-resolves trust_remote_code and rejects their custom tokenizer code. The
# deploy-config's trust_remote_code:true covers model loading but NOT that path.
# Higgs-Audio V3's README also prescribes --trust-remote-code.
extra_serve_args() {  # key -> extra serve flags (or empty)
    case "$1" in
        glm-tts|indextts2|voxcpm2|voxcpm2-reftext|higgs-audio-v3) echo "--trust-remote-code" ;;
        *)                                        echo "" ;;
    esac
}

# Extra environment variables a model's server needs, as a space-separated list of
# NAME=VALUE pairs exported before the server starts (the run drivers eval this).
# Higgs-Audio V3's README requires DeepGEMM disabled for correct/stable serving.
# Drama 3's deploy config (platforms.cuda.cuda_mps) starts a private MPS daemon per GPU;
# vllm-omni treats an inherited CUDA_MPS_PIPE_DIRECTORY (the slurm jobs' daemon-less
# workaround dir) as an operator-managed daemon and fails at startup, so clear it.
extra_env() {  # key -> "NAME=VALUE NAME=VALUE ..." (or empty)
    case "$1" in
        higgs-audio-v3) echo "VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0" ;;
        drama-3)        echo "CUDA_MPS_PIPE_DIRECTORY=" ;;
        *)              echo "" ;;
    esac
}

# Extra generate_omni.py arguments a model's server needs. Breeze TTS 2 infers the
# language from the text and rejects requests that carry a 'language' field.
extra_client_args() {  # key -> extra generate_omni.py args (or empty)
    case "$1" in
        breeze-tts-2) echo "--no-language" ;;
        *)            echo "" ;;
    esac
}
