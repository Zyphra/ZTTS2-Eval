# scripts/gen/lib_gen.sh -- shared helpers for the scripts/gen/run_*.sh drivers.
#
# SOURCE this from the repo root (`. scripts/gen/lib_gen.sh`); do not execute it. A failed
# source is the drivers' "not run from the repo root" check.
#
# Provides (plus repo_py from utils/lib_common.sh):
#   gen_venv_py VAR DEFAULT_DIR HINT     PY = ${!VAR:-DEFAULT_DIR}/bin/python, else exit 1
#   gen_langs TASK TEST_SET [SUPPORTED]  the languages to generate (stdout); GEN_NAME
#                                        labels the "[skip] unsupported" lines
#   gen_maybe_list LIST_FLAG NAMES       --list-langs true: print NAMES and exit 0
#   gen_nothing_to_do NAMES              empty NAMES: say so and exit 0
#   gen_require_ran RAN NAMES TASK       exit 1 when no language was generated
#   gen_done OUT_ROOT TASK LANGS         print the output dir and the scoring command
#   gen_default_out_root DRIVER MODEL [MODE]  a driver's default --out-root
#
# Every driver resolves its languages (gen_langs -> gen_maybe_list -> gen_nothing_to_do)
# BEFORE it checks its venv or server, so `--list-langs true` works on any host.

. utils/lib_common.sh

gen_venv_py() {  # <var> <default_dir> <hint>
    local dir=${!1:-$2}
    PY="${dir}/bin/python"
    if [ ! -x "${PY}" ]; then
        echo "error: no venv at ${dir} -- $3 (or set $1)" 1>&2
        exit 1
    fi
}

# Requested languages: TEST_SET, or the task's frozen basket (utils/benchmark.py) when
# empty. With SUPPORTED, the requested languages outside it are reported on stderr and
# dropped; the rest keep the requested order.
gen_langs() {  # <task> <test_set> [supported]
    local task=$1 test_set=$2 supported=${3:-} basket requested out="" l s hit
    if ! basket=$(python3 -m utils.benchmark langs "${task}" 2>/dev/null) || [ -z "${basket}" ]; then
        echo "error: unknown task '${task}'" 1>&2
        return 1
    fi
    requested=${test_set:-${basket}}
    if [ -z "${supported}" ]; then
        echo ${requested}
        return 0
    fi
    for l in ${requested}; do
        hit=0
        for s in ${supported}; do
            [ "${l}" = "${s}" ] && { hit=1; break; }
        done
        if [ "${hit}" -eq 1 ]; then
            out="${out:+${out} }${l}"
        else
            echo "[skip] ${l}: unsupported by ${GEN_NAME:-this driver}" 1>&2
        fi
    done
    echo "${out}"
}

gen_maybe_list() {  # <list_flag> <names>
    if [ "$1" = true ]; then
        echo "$2"
        exit 0
    fi
}

gen_nothing_to_do() {  # <names>
    if [ -z "${1// }" ]; then
        echo "[gen] nothing to generate: no requested language is supported"
        exit 0
    fi
}

gen_require_ran() {  # <ran> <names> <task>
    if [ "$1" -eq 0 ]; then
        echo "error: nothing generated (no data/$3/<lang>/text for: $2)" 1>&2
        exit 1
    fi
}

gen_done() {  # <out_root> <task> <langs>
    echo
    echo "generation done -> $1/$2/"
    echo "score with:  bash scripts/score/run_score.sh --decode-dir $1/$2 --test-set \"$3\""
}

# The default results root of each driver (scripts/gen/run_<DRIVER>.sh). MODE is the
# driver's output-splitting option where it has one: cartesia --mode, inworld --clone,
# zonos2 --accurate-mode, zonos3 --speaker-mode.
gen_default_out_root() {  # <driver> <model> [mode]
    local driver=$1 model=$2 mode=${3:-}
    case "${driver}" in
        audio8) echo "results/audio8-tts-realistic-0.6b" ;;
        cartesia)
            if [ "${mode}" = clone ]; then echo "results/cartesia_${model}_clone"
            else echo "results/cartesia_${model}"; fi ;;
        cosyvoice) echo "results/cosyvoice3" ;;
        elevenlabs) echo "results/elevenlabs_${model}" ;;
        elevenlabs_clone) echo "results/elevenlabs_${model}_clone" ;;
        gemini) echo "results/gemini_${model}" ;;
        inworld)
            if [ "${mode}" = true ]; then echo "results/inworld_${model}_clone"
            else echo "results/inworld_${model}"; fi ;;
        kokoro) echo "results/kokoro" ;;
        ming3|moss_tts|phonon|pocket_tts|zonos3_reftext) echo "results/${model}" ;;
        neutts) echo "results/neutts-2e" ;;
        omni) echo "results/omni_${model}" ;;
        vibevoice) echo "results/vibevoice-rt-0.5b" ;;
        vibevoice15) echo "results/vibevoice-${model}" ;;
        zonos2)
            case "${mode}" in
                true) echo "results/zonos2_accurate" ;;
                false) echo "results/zonos2_expressive" ;;
                *) echo "results/zonos2" ;;
            esac ;;
        zonos3) echo "results/zonos3-mini_${mode}" ;;
        *) echo "gen_default_out_root: unknown driver '${driver}'" 1>&2; return 1 ;;
    esac
}
