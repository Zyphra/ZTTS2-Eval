#!/usr/bin/env bash
# Generate Zonos2 audio against the VoxBlink2-derived eval (data/zero_shot_voxblink),
# starting/stopping the TTS server itself (like run_generate_omni_voxblink.sh). By
# default it launches the upstream Zonos2 Mini-SGLang server (scripts/serve_zonos2.sh)
# on a local port; pass --server <URL> to talk to an external one instead.
#
#   bash run_generate_zonos2_voxblink.sh                       # 9 zs-overlap langs
#   bash run_generate_zonos2_voxblink.sh --test_set "en zh ja" --gpu 0
#   bash run_generate_zonos2_voxblink.sh --test_set en --limit 3
#   bash run_generate_zonos2_voxblink.sh --accurate_mode false # expressive mode
#   bash run_generate_zonos2_voxblink.sh --gen_args "--clean-speaker-background"
#
# Score with: DECODE_DIR=results/zonos2/zero_shot_voxblink \
#             bash run_infer_voxblink_eval.sh
set -e

if [ ! -f utils/parse_options.sh ]; then
    echo "error: run this from the repo root" 1>&2; exit 1
fi

# Empty --server = self-manage (launch our own Zonos2 server); non-empty = external.
server=""
gpu=0
port=1919
model_path=
# Default to the 9 langs shared with zero_shot; the noisy voxblink-only tail
# (pt ar hi tr tl pl id th nl uk vi) is off by default. Re-enable via the full list or
# an explicit --test_set.
#   full 20: "en zh de es fr it ja ko ru pt ar hi tr tl pl id th nl uk vi"
test_set="en zh de es fr it ja ko ru"
out_root="results/zonos2/zero_shot_voxblink"
workers=16     # zonos2 mini-sglang handles many concurrent requests; bump for ~1.6k-prompt run
limit=0
stratified_bins=0    # if >0, sample K evenly-spaced utts per lang (voxblink is
                     # sorted worst->best MSR-UTMOS in 12 bins of ~10; --limit picks
                     # the worst N, --stratified_bins K picks one per bin)
overwrite=false
# Defaults match the released Zonos2 model's reference eval config: speaking-rate
# OFF, text normalization OFF (both opt-in). Bucket = index into /tts/capabilities
# speaking_rate_buckets (current server: 8 buckets; 3 = brisk-neutral 14-17).
speaking_rate_bucket=
text_norm=false
# Accuracy-mode token: server default is accurate ON; "false" = expressive
# mode, "true" = explicit accurate, empty = server default. Capability-gated;
# see run_generate_zonos2.sh.
accurate_mode=
# Raw passthrough to scripts/generate_zonos2.py, for sampling/conditioning sweeps
# (e.g. --gen_args "--clean-speaker-background --quality-buckets estimated_snr=11").
# Empty = the script's defaults, which are the released model's reference config.
# NOTE utils/parse_options.sh does `eval $name=\"$2\"`, so quoted JSON inside this
# string loses its inner quotes -- use --quality-buckets' quote-free k=v:k=v form.
gen_args=

. utils/parse_options.sh || exit 1

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi

# --- server lifecycle: empty --server launches our own (like the omni voxblink
# driver); non-empty --server uses that external one, unmanaged.
caps_json=$(mktemp /tmp/zonos2_caps.XXXXXX.json)
if [ -z "${server}" ]; then
    export ZONOS2_HOST=127.0.0.1
    export ZONOS2_PORT="${port}"
    export ZONOS2_GPU="${gpu}"
    [ -n "${model_path}" ] && export ZONOS2_MODEL="${model_path}"
    . scripts/serve_zonos2.sh
    trap 'rm -f "${caps_json}"; stop_zonos2_server' EXIT INT TERM
    if ! start_zonos2_server; then
        echo "error: failed to launch Zonos2 server (see log above)" 1>&2; exit 1
    fi
    server="${ZONOS2_BASE_URL}"
else
    trap 'rm -f "${caps_json}"' EXIT
    echo "[zonos2] using external server ${server} (not managed here)"
fi

# Check the HTTP status (a plain `curl -s` exits 0 on a 404/500 body) and
# validate the JSON once, loudly, before the quiet .get() probes below
# (same pattern as run_generate_zonos2.sh).
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
fi

# The benchmark task is zero-shot cloning; a server without speaker support
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

# Voxblink lang (ISO 639-1) -> ZONOS2 text-normalization code (VALID TN codes only:
# en_us en_gb fr_fr de es it pt_br ja cmn ko). `language` selects TN only; the model
# byte-synthesises ANY language. Langs with no TN code (ru pl tr hi id th ar tl ...)
# map to "" -> generate_zonos2.py's default en_us is used (passes validation; with TN
# off, no effect on the byte-synthesised audio). No language is skipped.
lang_for() {
    case "$1" in
        en) echo en_us ;;
        zh) echo cmn ;;
        de) echo de ;;
        es) echo es ;;
        fr) echo fr_fr ;;
        it) echo it ;;
        ja) echo ja ;;
        ko) echo ko ;;
        pt) echo pt_br ;;
        *)  echo "" ;;             # no TN code -> default en_us, byte-synthesised
    esac
}

# Server-side text normalization: the current server's normalizer covers
# {en_us en_gb fr_fr de es it pt_br ja cmn ko} per /tts/capabilities (no ru).
# ja/ko/pt advertised but not yet validated here -- unmapped until checked.
norm_lang_for() {
    case "$1" in
        en) echo en ;; zh) echo zh ;; de) echo de ;; es) echo es ;;
        fr) echo fr ;; it) echo it ;;
        *)  echo "" ;;
    esac
}

for nm in ${test_set}; do
    data_dir="data/zero_shot_voxblink/${nm}"
    out_dir="${out_root}/${nm}"
    if [ ! -f "${data_dir}/text" ]; then
        echo "[skip] ${data_dir}: no 'text' file" 1>&2; continue
    fi
    lg="$(lang_for ${nm})"
    # Empty lg -> omit --language (default en_us); byte mode synthesises the real lang.
    lang_opt=""; [ -n "${lg}" ] && lang_opt="--language ${lg}"
    nl="$(norm_lang_for ${nm})"
    norm_opt=""; [ "${text_norm}" = "true" ] && [ -n "${nl}" ] && norm_opt="--text-normalization --normalizer-language ${nl}"
    ow_opt=""; [ "${overwrite}" = "true" ] && ow_opt="--overwrite"

    strat_opt=""; [ "${stratified_bins}" -gt 0 ] && strat_opt="--stratified-bins ${stratified_bins}"
    echo "=== [zonos2] ${data_dir} -> ${out_dir}  ${lang_opt} ${norm_opt} ${sr_opt} ${acc_opt} ${strat_opt} ==="
    "${PY}" scripts/generate_zonos2.py \
        --server "${server}" \
        --data-dir "${data_dir}" \
        --out-dir "${out_dir}" \
        --workers "${workers}" \
        --limit "${limit}" \
        ${lang_opt} ${norm_opt} ${sr_opt} ${acc_opt} ${strat_opt} ${ow_opt} ${gen_args}
done

echo
echo "generation done -> ${out_root}/"
echo "score with: DECODE_DIR=${out_root} bash run_infer_voxblink_eval.sh"
