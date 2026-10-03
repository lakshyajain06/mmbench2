#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <ms|mw> <gpu-index> [output-directory]" >&2
    exit 2
fi

domain="$1"
gpu="$2"
case "$domain" in
    ms|mw) ;;
    *) echo "domain must be 'ms' or 'mw'" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
config="$repo_root/evaluation/configs/random_start_msmw_val.json"
output="${3:-$repo_root/evaluation/runs/random_start_msmw_val_${domain}}"
python_bin="${PYTHON_BIN:-python}"

if [[ -s "$output/windows.jsonl" || -s "$output/steps.jsonl" ]]; then
    echo "refusing to append to existing result files in $output" >&2
    exit 1
fi

mapfile -t tasks < <(
    "$python_bin" -c \
        'import json,sys; c=json.load(open(sys.argv[1])); p=sys.argv[2]+"-"; print(*[t for t in c["tasks"]["first_pass"] if t.startswith(p)], sep="\n")' \
        "$config" "$domain"
)

cd "$repo_root"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" evaluation/cli.py \
    --config "$config" \
    --set "output=$("$python_bin" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$output")" \
    run --partitions val --tasks "${tasks[@]}"
