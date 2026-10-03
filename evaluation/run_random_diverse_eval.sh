#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <regular|finetuned> <ms|mw> <gpu-index> [output-directory [task ...]]" >&2
    exit 2
fi

checkpoint="$1"
domain="$2"
gpu="$3"
case "$checkpoint" in
    regular|finetuned) ;;
    *) echo "checkpoint must be 'regular' or 'finetuned'" >&2; exit 2 ;;
esac
case "$domain" in
    ms|mw) ;;
    *) echo "domain must be 'ms' or 'mw'" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
config="$repo_root/evaluation/configs/random_start_diverse_30_val.json"
output="${4:-$repo_root/evaluation/runs/random_diverse30_${checkpoint}_${domain}}"
if [[ "$output" != /* ]]; then
    output="$repo_root/$output"
fi
python_bin="${PYTHON_BIN:-/home_shared/grail_lakshya/miniforge3/envs/mmbench2/bin/python}"

if [[ "${ALLOW_RESUME:-0}" != "1" && ( -s "$output/windows.jsonl" || -s "$output/steps.jsonl" ) ]]; then
    echo "refusing to append to existing result files in $output" >&2
    exit 1
fi

if [[ $# -gt 4 ]]; then
    tasks=("${@:5}")
else
    mapfile -t tasks < <(
        "$python_bin" -c \
            'import json,sys; c=json.load(open(sys.argv[1])); p=sys.argv[2]+"-"; print(*[t for t in c["tasks"]["first_pass"] if t.startswith(p)], sep="\n")' \
            "$config" "$domain"
    )
fi

overrides=(--set "output=$("$python_bin" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$output")")
if [[ "$checkpoint" == "finetuned" ]]; then
    dynamics="$repo_root/mmbench2-finetuned/dynamics.pt"
    overrides+=(--set "model.dynamics_ckpt=$("$python_bin" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$dynamics")")
fi

cd "$repo_root"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" evaluation/cli.py \
    --config "$config" "${overrides[@]}" \
    run --partitions val --tasks "${tasks[@]}"
