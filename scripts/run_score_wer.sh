input_text=$(readlink -f "$1")
output_dir=$(readlink -f "$2")
lang=$3

# cal_wer.sh resolves its sibling python scripts relative to utils/, so run it from there.
( cd "$(dirname "$0")/../utils" && bash cal_wer.sh "${input_text}" "${output_dir}" "${lang}" )
