# scripts/gen/lib_gen.sh -- shared helpers for the scripts/gen/run_*.sh drivers.
#
# SOURCE this from the repo root (`. scripts/gen/lib_gen.sh`); do not execute it. A failed
# source is the drivers' "not run from the repo root" check.
#
# Provides (plus repo_py from utils/lib_common.sh):
#   gen_langs TASK TEST_SET [SUPPORTED] the language list to generate (stdout)
#   gen_require_ran RAN NAMES TASK     exit 1 when no language was generated
#   gen_done OUT_ROOT TASK LANGS       print the output dir and the scoring command

. utils/lib_common.sh

# An explicit TEST_SET is echoed unchanged (each driver keeps its own policy for
# languages it does not support). Empty TEST_SET -> the task's frozen basket
# (utils/benchmark.py), filtered to SUPPORTED when given, basket order kept.
gen_langs() {  # <task> <test_set> [supported]
    local task=$1 test_set=$2 supported=${3:-} basket out="" l s
    if [ -n "${test_set}" ]; then
        echo "${test_set}"
        return 0
    fi
    if ! basket=$(python3 -m utils.benchmark langs "${task}" 2>/dev/null) || [ -z "${basket}" ]; then
        echo "error: unknown task '${task}'" 1>&2
        return 1
    fi
    if [ -z "${supported}" ]; then
        echo "${basket}"
        return 0
    fi
    for l in ${basket}; do
        for s in ${supported}; do
            [ "${l}" = "${s}" ] && { out="${out:+${out} }${l}"; break; }
        done
    done
    echo "${out}"
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
    echo "score with:  bash scripts/score/run_score.sh --decode_dir $1/$2 --test_set \"$3\""
}
