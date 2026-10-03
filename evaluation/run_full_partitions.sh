#!/usr/bin/env bash
set -u
if [[ $# -ne 4 ]]; then
  echo 'usage: run_full_partitions.sh GPU WAIT_PID OUTPUT PARTITIONS_CSV' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
wait_pid="$2"
output="$3"
IFS=',' read -r -a partitions <<< "$4"
python_bin=/home_shared/grail_lakshya/miniforge3/envs/mmbench2/bin/python
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$script_dir")" || exit 2
mkdir -p "$output"
while ps -p "$wait_pid" -o args= 2>/dev/null | rg -q 'sampling.episodes_per_task=2'; do
  sleep 30
done
printf 'full run started %s on GPU %s for %s\n' "$(date -u)" "$CUDA_VISIBLE_DEVICES" "$4" | tee -a "$output/full.log"
for attempt in 1 2 3 4 5; do
  "$python_bin" evaluation/cli.py --set "output=\"$output\"" run --metrics-only-test --partitions "${partitions[@]}" >> "$output/full.log" 2>&1
  status=$?
  if [[ "$status" -eq 0 ]]; then
    printf '0\n' > "$output/full.exit"
    printf 'full run finished %s\n' "$(date -u)" | tee -a "$output/full.log"
    exit 0
  fi
  printf 'attempt %s failed with %s at %s\n' "$attempt" "$status" "$(date -u)" | tee -a "$output/full.log"
  sleep 30
done
printf '%s\n' "$status" > "$output/full.exit"
exit "$status"
