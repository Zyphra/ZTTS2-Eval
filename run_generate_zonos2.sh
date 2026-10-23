#!/usr/bin/env bash
# Generate Zonos2 audio, starting/stopping the TTS server itself -- just like
# run_generate_omni.sh does for vLLM-Omni. By default this launches the upstream
# Zonos2 Mini-SGLang server (scripts/serve_zonos2.sh -> `python -m zonos2`) on a
# local port, generates through it, then stops it. Pass --server <URL> to instead
# talk to an already-running external server (the old behaviour; no launch/stop).
#
# For each task/test_set this:
#   - resolves data/<task>/<nm>/text + prompt_wav.scp
#   - calls scripts/generate_zonos2.py against the server
#   - writes wavs to <out_root>/<task>/<nm>/wavs/
#
# Score afterwards with run_infer_zonos2_eval.sh, pointing DECODE_DIR at the
# matching <out_root>/<task>/ directory.
#
#   bash run_generate_zonos2.sh                        # all sets, self-managed server
#   bash run_generate_zonos2.sh --task zero_shot --gpu 0
#   bash run_generate_zonos2.sh --task zero_shot --test_set en --limit 3
#   bash run_generate_zonos2.sh --accurate_mode false # expressive mode
#   bash run_generate_zonos2.sh --server http://host:1919   # external server
#
# Options (Kaldi-style "--name value", parsed by utils/parse_options.sh):
#   --server     external TTS endpoint; EMPTY = launch our own  (default '' = self-managed)
#   --gpu        GPU index for the launched server              (default 0)
#   --port       port for the launched server                   (default 1919)
#   --model_path model path/id for the launched server          (default serve default)
#   --task       restrict to one task group                     (optional)
#   --test_set   restrict to names within --task                (optional; needs --task)
#   --out_root   results root                                   (default results/zonos2)
#   --workers    concurrent requests per test set               (default 4)
#   --limit      only the first N utterances per set            (default 0 = all)
#   --overwrite  regenerate existing wavs                       (default false)
#   --gen_args   raw passthrough to scripts/generate_zonos2.py  (default '')
#                e.g. --gen_args "--temperature 0.7 --topk 90"
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root (utils/parse_options.sh not found)" 1>&2
    exit 1
fi

# Empty --server = self-manage (launch our own Zonos2 server); non-empty = external.
server=""
gpu=0
port=1919
model_path=
task=
test_set=
out_root="results/zonos2"
workers=4
limit=0
overwrite=false
# Raw passthrough to scripts/generate_zonos2.py, for sampling/decoding sweeps
# (e.g. --gen_args "--temperature 0.7 --topk 90 --repetition-window 128").
# Empty = the script's defaults, which are the released model's reference config.
gen_args=
# Defaults below match the released Zonos2 model's reference eval config:
# speaking-rate OFF, text normalization OFF. Both remain opt-in.
#
# Speaking-rate bucket -- index into /tts/capabilities speaking_rate_buckets
# (current server: 8 buckets, 0-8 8-11 11-14 14-17 17-21 21-28 28-40 40+;
# 3 = brisk-neutral 14-17). Empty = disabled.
speaking_rate_bucket=
# Server-side text normalization: "true" enables the per-language mapping in
# norm_lang_for below.
text_norm=false
# Accuracy-mode token (verified against the zonos2 server source): the server
# DEFAULTS to accurate mode ON; "false" switches to expressive mode, "true"
# sends accurate explicitly, empty leaves the server default. Gated on the
# accurate_mode_token_enabled capability -- the script aborts if set on a
# server without it (unknown fields are silently ignored, which would
# otherwise fake the eval).
accurate_mode=

help_message="usage: bash run_generate_zonos2.sh [--server URL|''] [--gpu N] [--port N] [--model_path M] [--task T] [--test_set \"...\"] [--out_root D] [--workers N] [--limit N] [--overwrite true] [--gen_args \"...\"] [--speaking_rate_bucket N|''] [--text_norm true] [--accurate_mode true]"
. utils/parse_options.sh || exit 1

if [ -n "${test_set}" ] && [ -z "${task}" ]; then
    echo "error: --test_set requires --task" 1>&2
    exit 1
fi

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

# --- server lifecycle -----------------------------------------------------------
# Empty --server: launch our OWN Zonos2 server (like run_generate_omni.sh), stop it
# on exit. Non-empty --server: use that external server, don't manage its lifecycle.
caps_json=$(mktemp /tmp/zonos2_caps.XXXXXX.json)
if [ -z "${server}" ]; then
    export ZONOS2_HOST=127.0.0.1
    export ZONOS2_PORT="${port}"
    export ZONOS2_GPU="${gpu}"
    [ -n "${model_path}" ] && export ZONOS2_MODEL="${model_path}"
    . scripts/serve_zonos2.sh
    # Clean up both the caps file and the server we started, on any exit.
    trap 'rm -f "${caps_json}"; stop_zonos2_server' EXIT INT TERM
    if ! start_zonos2_server; then
        echo "error: failed to launch Zonos2 server (see log above)" 1>&2
        exit 1
    fi
    server="${ZONOS2_BASE_URL}"
else
    trap 'rm -f "${caps_json}"' EXIT
    echo "[zonos2] using external server ${server} (not managed here)"
fi

# Quick server health check -- fail fast with a clear message rather than 500ing
# every request. /tts/capabilities is one of the cheapest endpoints; we also
# read it to gate the speaking-rate and speaker (voice-cloning) flags.
# Check the HTTP status (a plain `curl -s` exits 0 on a 404/500 body) and
# validate the JSON once, loudly, before the quiet .get() probes below.
http_code=$(curl -s -m 5 -o "${caps_json}" -w '%{http_code}' "${server}/tts/capabilities" || true)
if [ "${http_code}" != "200" ]; then
    echo "error: Zonos2 server at ${server} not healthy (GET /tts/capabilities -> HTTP ${http_code:-<no response>})" 1>&2
    if [ -s "${caps_json}" ]; then
        echo "response body (first 400 bytes):" 1>&2
        head -c 400 "${caps_json}" 1>&2; echo 1>&2
    fi
    exit 1
fi
if ! "${PY}" -c "import json; json.load(open('${caps_json}'))"; then
    echo "error: ${server}/tts/capabilities returned invalid JSON (first 400 bytes):" 1>&2
    head -c 400 "${caps_json}" 1>&2; echo 1>&2
    exit 1
fi
sr_supported=$(${PY} -c "import json; print(json.load(open('${caps_json}')).get('speaking_rate_enabled', False))" 2>/dev/null)
sr_opt=""
if [ "${sr_supported}" = "True" ] && [ -n "${speaking_rate_bucket}" ]; then
    sr_opt="--speaking-rate-bucket ${speaking_rate_bucket}"
    echo "[zonos2] speaking_rate_bucket=${speaking_rate_bucket} (server supports it)"
elif [ -n "${speaking_rate_bucket}" ]; then
    echo "[zonos2] speaking_rate_bucket requested (${speaking_rate_bucket}) but server does not expose speaking_rate_enabled -- skipping"
fi

# The benchmark task is zero-shot cloning, so a server without speaker support
# is not usable here -- fail fast instead of silently generating default voices.
spk_enabled=$(${PY} -c "import json; print(json.load(open('${caps_json}')).get('speaker_enabled', True))" 2>/dev/null)
if [ "${spk_enabled}" = "False" ]; then
    echo "error: server reports speaker_enabled=false (no voice cloning) -- not usable for this benchmark" 1>&2
    exit 1
fi

acc_opt=""
if [ -n "${accurate_mode}" ]; then
    acc_supported=$(${PY} -c "import json; print(json.load(open('${caps_json}')).get('accurate_mode_token_enabled', False))" 2>/dev/null)
    if [ "${acc_supported}" != "True" ]; then
        echo "error: --accurate_mode set but server does not expose accurate_mode_token_enabled" 1>&2
        exit 1
    fi
    case "${accurate_mode}" in
        true)  acc_opt="--accurate-mode" ;;
        false) acc_opt="--no-accurate-mode" ;;
        *) echo "error: --accurate_mode must be true, false, or empty" 1>&2; exit 1 ;;
    esac
    echo "[zonos2] accurate_mode=${accurate_mode} (server supports the token)"
fi

# --- task / test-set registry --------------------------------------------------
# Sets are the same as run_generate_omni.sh PLUS the FLEURS-R languages we
# rebuilt (de es fr it ru) -- they exist under data/ and we want to score them.
sets_for() {
    case "$1" in
        zero_shot)              echo "en zh de es fr it ja ko ru" ;;
        *) echo "" ;;
    esac
}

# `language` selects TEXT NORMALIZATION only (off by default here); in byte mode the
# model synthesises the raw UTF-8 text for ANY language. But the server validates
# `language` against its TN-language list (SERVER_TO_NEMO_LANG: en_us en_gb fr_fr de
# es it pt_br ja cmn ko) and 400s on anything else -- so map to a VALID TN code where
# one exists, and leave the rest empty (ru etc.): generate_zonos2.py then falls back
# to its default en_us, which passes validation and, with TN off, does not affect the
# synthesised (byte-tokenised) audio. No language is skipped.
lang_for() {
    case "$1/$2" in
        zero_shot/en) echo en_us ;;
        zero_shot/zh) echo cmn ;;
        zero_shot/de) echo de ;;
        zero_shot/es) echo es ;;
        zero_shot/fr) echo fr_fr ;;
        zero_shot/it) echo it ;;
        zero_shot/ja) echo ja ;;
        zero_shot/ko) echo ko ;;
        zero_shot/ru) echo "" ;;   # no TN code -> default en_us, byte-synthesised
        *) echo "" ;;
    esac
}

# Server-side text normalization config. Per /tts/capabilities the current
# server's normalizer covers {en_us en_gb fr_fr de es it pt_br ja cmn ko} -- no
# ru. ja/ko are advertised but not yet validated here, so they stay unmapped
# (no norm) until checked.
norm_lang_for() {
    case "$1/$2" in
        zero_shot/en) echo en ;;
        zero_shot/zh) echo zh ;;
        zero_shot/de) echo de ;;
        zero_shot/es) echo es ;;
        zero_shot/fr) echo fr ;;
        zero_shot/it) echo it ;;
        *) echo "" ;;            # ja, ko, ru -> no norm
    esac
}

tasks="${task:-zero_shot}"

for tk in ${tasks}; do
    names="${test_set:-$(sets_for ${tk})}"
    if [ -z "${names}" ]; then
        echo "error: unknown task '${tk}'" 1>&2
        exit 1
    fi
    for nm in ${names}; do
        data_dir="data/${tk}/${nm}"
        out_dir="${out_root}/${tk}/${nm}"
        if [ ! -f "${data_dir}/text" ]; then
            echo "[skip] ${data_dir}: no 'text' file" 1>&2
            continue
        fi

        lg="$(lang_for ${tk} ${nm})"
        # Empty lg (no TN code) -> omit --language so generate_zonos2.py's default
        # en_us is used; in byte mode with TN off this synthesises the real language.
        lang_opt=""; [ -n "${lg}" ] && lang_opt="--language ${lg}"

        nl="$(norm_lang_for ${tk} ${nm})"
        norm_opt=""
        [ "${text_norm}" = "true" ] && [ -n "${nl}" ] && norm_opt="--text-normalization --normalizer-language ${nl}"

        ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

        echo "=== [zonos2] ${data_dir} -> ${out_dir}  ${lang_opt} ${norm_opt} ${sr_opt} ${acc_opt} ==="
        "${PY}" scripts/generate_zonos2.py \
            --server "${server}" \
            --data-dir "${data_dir}" \
            --out-dir "${out_dir}" \
            --workers "${workers}" \
            --limit "${limit}" \
            ${lang_opt} ${norm_opt} ${sr_opt} ${acc_opt} ${ow_opt} ${gen_args}
    done
done

echo
echo "generation done -> ${out_root}/"
echo "score zero_shot with:  DECODE_DIR=${out_root}/zero_shot bash run_infer_zonos2_eval.sh --task zero_shot --test_set \"$(sets_for zero_shot)\""
