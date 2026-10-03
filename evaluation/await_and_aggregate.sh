#!/usr/bin/env bash
set -u
root="$(cd "$(dirname "$0")" && pwd)/runs"
a="$root/partition_eval_a"
b="$root/partition_eval_b"
combined="$root/combined_full"
mkdir -p "$combined"
while [[ ! -f "$a/full.exit" || ! -f "$b/full.exit" ]]; do
  sleep 60
done
if [[ "$(cat "$a/full.exit")" != 0 || "$(cat "$b/full.exit")" != 0 ]]; then
  printf 'one or more evaluation workers failed\n' > "$combined/FAILED"
  exit 1
fi
python "$(dirname "$0")/aggregate_metrics.py" --runs "$a" "$b" \
  --output "$combined" --expected-windows 38400 > "$combined/aggregate.log" 2>&1
status=$?
printf '%s\n' "$status" > "$combined/aggregate.exit"
exit "$status"
