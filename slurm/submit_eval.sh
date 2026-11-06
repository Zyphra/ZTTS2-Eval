#!/usr/bin/env bash
# slurm/submit_eval.sh -- submit local-model eval runs to SLURM, ONE job per model.
#
# Fans the model sweep across GPUs/nodes instead of cycling models sequentially on a
# single GPU: for each selected model it `sbatch`es slurm/eval_job.sh, which generates
# then scores that model inside one 1-GPU allocation.
#
# Run from the repo root.
#
#   # all omni models, one 1-GPU job each, gen + score:
#   bash slurm/submit_eval.sh --engine omni --model all
#
#   # a single omni model, one language, quick smoke test:
#   bash slurm/submit_eval.sh --engine omni --model qwen3-tts-0.6b \
#        --task clean --test_set en --limit 3
#
#   # zonos2, self-managed server, both modes (accurate="stability" + expressive):
#   bash slurm/submit_eval.sh --engine zonos2 --accurate_mode true  --task clean
#   bash slurm/submit_eval.sh --engine zonos2 --accurate_mode false --task clean
#
#   # zonos2 against an already-running external server instead of self-managing:
#   bash slurm/submit_eval.sh --engine zonos2 --server http://<host>:<port> \
#        --task clean
#
#   # see the sbatch commands without submitting:
#   bash slurm/submit_eval.sh --engine omni --model all --dry-run true
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --engine      omni|zonos2|kokoro|cosyvoice3|moss-tts|neutts|vibevoice|
#                 vibevoice15|audio8                                (default omni)
#   --model       omni model key or "all"; ignored for zonos2     (default all)
#   --task        clean | in-the-wild                  (default clean)
#   --test_set    space-separated langs (quote it), '' = default  (default '')
#   --workers     concurrent generation requests                  (default 16)
#   --limit       first N utts per set, 0 = all                   (default 0)
#   --score       run WER/spk/MSR-UTMOS after generation          (default true)
#   --overwrite   regenerate existing wavs (passed to the gen)     (default false)
#   --server      external zonos2 URL; EMPTY = self-manage (zonos2 only)  (default empty)
#   --accurate_mode  zonos2 mode: true=accurate ("stability"), false=expressive,
#                 empty=server default (zonos2 only)              (default empty)
#   --model_path  zonos2 model path/id for the launched server    (default serve default)
#   --partition   SLURM partition                                 (default: cluster default)
#   --account     SLURM account                                   (default: cluster default)
#   --time        SLURM wall-clock limit; empty = 24h              (default '')
#   --gpus        GPUs per job                                     (default 1)
#   --cpus        CPUs per job (--cpus-per-task)                   (default 28)
#   --mem         memory per job                                  (default 200G)
#   --exclude     nodes to avoid (sbatch --exclude), e.g. a node   (default '')
#                 hogged by non-SLURM jobs
#   --nodelist    force specific node(s) (sbatch --nodelist)       (default '')
#   --dependency  sbatch --dependency spec, e.g. afterany:123:124   (default '')
#                 Use this to SERIALIZE jobs. Concurrent jobs packed onto one node
#                 OOM each other during the scoring phase (Qwen3-ASR + Whisper are
#                 co-resident). The run_infer scripts now exit non-zero on scoring
#                 failures (the job goes red instead of silently exiting 0), but
#                 serializing still avoids the wasted reruns.
#   --dry-run     print the sbatch commands, don't submit         (default false)
#
# zonos2 server: by default each zonos2 job now LAUNCHES its own Zonos2 Mini-SGLang
# server on the allocated GPU (scripts/serve_zonos2.sh), generates through it, stops
# it, then reuses the GPU for scoring -- the same self-managed shape as the omni
# engine. The server code lives in a separate checkout (set ZONOS2_HOME to a
# ZONOS2 checkout) with its own venv; override via ZONOS2_* env or
# --model_path. Pass --server <URL> to reuse an external server instead of launching.
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

engine=omni
model=all
task=clean
test_set=
workers=16
limit=0
score=true
overwrite=false
# zonos2 only. server EMPTY = the job self-manages its own Zonos2 server (like the
# omni engine); set it to an external URL to reuse an already-running one instead.
server=""
# zonos2 accurate_mode: true = accurate ("stability"), false = expressive, empty =
# server default. Routes output to results/zonos2_accurate | _expressive | zonos2.
accurate_mode=
model_path=
partition=
account=
# Empty = 24h (applied below); eval_job.sh's own #SBATCH --time=10:00:00 only applies to direct sbatch runs.
time=
gpus=1
cpus=28
mem=200G
exclude=
nodelist=
dependency=
dry_run=false

help_message="usage: bash slurm/submit_eval.sh --engine {omni|zonos2} [--model {all|<key>}] [--task T] [--test_set \"...\"] [--partition P] [--account A] [--dry-run true]"
. utils/parse_options.sh || exit 1

case "${task}" in
    clean|in-the-wild) ;;
    *) echo "error: --task must be clean or in-the-wild (got '${task}')" 1>&2; exit 1 ;;
esac

# Resolve the list of "models" (job units) for the chosen engine.
case "${engine}" in
    omni)
        # Single source of truth: scripts/omni_models.sh (the registry both gen
        # drivers source) provides ALL_MODELS / NEW_MODELS / model_spec. Validate
        # exactly like run_generate_omni.sh does: `--model all` sweeps ALL_MODELS,
        # and anything model_spec knows (NEW_MODELS, comparison variants like
        # omnivoice-reftext, retired keys) is submittable by explicit --model.
        . scripts/omni_models.sh
        if [ "${model}" = all ]; then models="${ALL_MODELS}"; else models="${model}"; fi
        for m in ${models}; do
            if [ -z "$(model_spec "${m}")" ]; then
                echo "error: unknown omni model '${m}'" 1>&2
                echo "known: all ${ALL_MODELS} ${NEW_MODELS} (see scripts/omni_models.sh)" 1>&2
                exit 1
            fi
        done
        ;;
    zonos2)
        # zonos2 is a single server-backed model; the job unit is just "zonos2".
        models="zonos2"
        [ "${model}" != all ] && [ "${model}" != zonos2 ] && \
            echo "[note] --model is ignored for --engine zonos2 (single model)" 1>&2
        ;;
    kokoro)
        # kokoro is a single in-process model (its own venv); the job unit is "kokoro".
        models="kokoro"
        [ "${model}" != all ] && [ "${model}" != kokoro ] && \
            echo "[note] --model is ignored for --engine kokoro (single model)" 1>&2
        ;;
    cosyvoice3)
        # cosyvoice3 native is a single in-process model (its own venv); unit "cosyvoice3".
        models="cosyvoice3"
        [ "${model}" != all ] && [ "${model}" != cosyvoice3 ] && \
            echo "[note] --model is ignored for --engine cosyvoice3 (single model)" 1>&2
        ;;
    moss-tts)
        # MOSS-TTS family (native): --model selects the variant, or "all" for every one.
        # moss-tts-nano is RETIRED (decode bug) -- run_generate_moss_tts.sh still knows
        # the variant, but it is out of the sweep. See scripts/omni_models.sh.
        ALL_MOSS="moss-tts moss-tts-v1.5 moss-tts-local moss-tts-local-v1.5"
        if [ "${model}" = all ]; then models="${ALL_MOSS}"; else models="${model}"; fi
        for m in ${models}; do
            case " ${ALL_MOSS} " in *" ${m} "*) ;; *)
                echo "error: unknown moss variant '${m}' (known: all ${ALL_MOSS})" 1>&2; exit 1 ;;
            esac
        done
        ;;
    neutts)
        # neutts native is a single in-process model (its own venv); unit "neutts".
        models="neutts"
        [ "${model}" != all ] && [ "${model}" != neutts ] && \
            echo "[note] --model is ignored for --engine neutts (single model)" 1>&2
        ;;
    vibevoice)
        # vibevoice native is a single in-process model (its own venv); unit "vibevoice".
        models="vibevoice"
        [ "${model}" != all ] && [ "${model}" != vibevoice ] && \
            echo "[note] --model is ignored for --engine vibevoice (single model)" 1>&2
        ;;
    vibevoice15)
        # VibeVoice-1.5B native (its own venv ~/.venvs/vibevoice15); unit "vibevoice15".
        # Distinct engine from `vibevoice` (the realtime 0.5B): different weights, a
        # different vendored code tree, and clone vs preset mode. See
        # third_party/VibeVoice-1.5B/PROVENANCE.md.
        # Two variants share this engine: 1.5b (microsoft/VibeVoice-1.5B) and 7b
        # (vibevoice/VibeVoice-7B, a third-party backup of the withdrawn Large model).
        ALL_VV15="1.5b 7b"
        if [ "${model}" = all ]; then models="${ALL_VV15}"
        elif [ "${model}" = vibevoice15 ]; then models="1.5b"
        else models="${model}"; fi
        for m in ${models}; do
            case " ${ALL_VV15} " in *" ${m} "*) ;; *)
                echo "error: unknown vibevoice15 variant '${m}' (known: all ${ALL_VV15})" 1>&2; exit 1 ;;
            esac
        done
        ;;
    audio8)
        # Audio8-TTS-Realistic-0.6B (SGLang-Omni adapter server, own venv); unit "audio8".
        models="audio8"
        [ "${model}" != all ] && [ "${model}" != audio8 ] && \
            echo "[note] --model is ignored for --engine audio8 (single model)" 1>&2
        ;;
    *)
        echo "error: --engine must be omni, zonos2, kokoro, cosyvoice3, moss-tts, neutts, vibevoice, vibevoice15 or audio8 (got '${engine}')" 1>&2; exit 1 ;;
esac

REPO_ROOT="$(pwd -P)"
mkdir -p logs/slurm

# Config shared by every job goes through the environment (sbatch --export=ALL). This
# avoids the comma/space escaping problems of `sbatch --export=VAR=value` -- notably
# for multi-word --test_set lists like "en zh de".
export EJ_REPO="${REPO_ROOT}"
export EJ_ENGINE="${engine}"
export EJ_TASK="${task}"
export EJ_TEST_SET="${test_set}"
export EJ_WORKERS="${workers}"
export EJ_LIMIT="${limit}"
export EJ_SCORE="${score}"
export EJ_OVERWRITE="${overwrite}"
export EJ_SERVER="${server}"
export EJ_ACCURATE_MODE="${accurate_mode}"
export EJ_MODEL_PATH="${model_path}"

echo "submitting ${engine} eval: task=${task} test_set='${test_set:-<default>}' score=${score}"
echo "  partition=${partition:-<cluster default>} account=${account:-<cluster default>} time=${time:-24:00:00} gpus=${gpus} cpus=${cpus} mem=${mem}"
echo "  models: ${models}"

for m in ${models}; do
    export EJ_MODEL="${m}"
    job_name="ztts-${engine}-${m}"
    # zonos2 runs one accurate_mode variant per job -- tag the job/log by mode so the
    # two modes (accurate="stability" vs expressive) don't share a name or log file.
    if [ "${engine}" = zonos2 ] && [ -n "${accurate_mode}" ]; then
        [ "${accurate_mode}" = true ] && job_name="ztts-zonos2-accurate" || job_name="ztts-zonos2-expressive"
    fi
    sbatch_cmd=(sbatch
        --job-name="${job_name}"
        --gpus="${gpus}"
        --cpus-per-task="${cpus}"
        --mem="${mem}"
        --chdir="${REPO_ROOT}"
        --output="logs/slurm/${job_name}-%j.out"
        --export=ALL)
    [ -n "${partition}" ] && sbatch_cmd+=(--partition="${partition}")
    [ -n "${account}" ] && sbatch_cmd+=(--account="${account}")
    # Time limit: --time if given, else 24h (eval_job.sh's 10h #SBATCH fallback is for direct runs).
    eff_time="${time}"
    [ -z "${eff_time}" ] && eff_time=24:00:00
    [ -n "${eff_time}" ] && sbatch_cmd+=(--time="${eff_time}")
    # Node steering (useful when a node is hogged by non-SLURM jobs).
    [ -n "${exclude}" ] && sbatch_cmd+=(--exclude="${exclude}")
    [ -n "${nodelist}" ] && sbatch_cmd+=(--nodelist="${nodelist}")
    [ -n "${dependency}" ] && sbatch_cmd+=(--dependency="${dependency}")
    sbatch_cmd+=(slurm/eval_job.sh)

    if [ "${dry_run}" = true ]; then
        echo "[dry-run] EJ_MODEL=${m} ${sbatch_cmd[*]}"
    else
        "${sbatch_cmd[@]}"
    fi
done

if [ "${dry_run}" != true ]; then
    echo
    echo "submitted. watch with:  squeue -u \$USER -o '%.18i %.9P %.28j %.8T %.10M %R'"
    echo "logs:                   logs/slurm/ztts-${engine}-<model>-<jobid>.out"
fi
