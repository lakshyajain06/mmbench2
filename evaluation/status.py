"""Show progress for the five-partition robotics checkpoint evaluation."""
import argparse
import collections
import json
from pathlib import Path

PARTITIONS = ('expert', 'mixed-small', 'mixed-large', 'val', 'test')
TASKS = 8
EPISODES = 20
HORIZONS = 4
STARTS = 4
SEEDS = 3
EXPECTED_PER_PARTITION = TASKS * EPISODES * HORIZONS * STARTS * SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path,
                        default=[Path(__file__).parent / 'runs/partition_eval_a',
                                 Path(__file__).parent / 'runs/partition_eval_b'])
    args = parser.parse_args()
    counts = collections.Counter()
    for directory in args.runs:
        path = directory / 'windows.jsonl'
        if path.exists():
            with path.open() as f:
                for line in f:
                    counts[json.loads(line)['partition']] += 1
        exit_path = directory / 'full.exit'
        state = ('exit ' + exit_path.read_text().strip()) if exit_path.exists() else 'running'
        print(f'{directory.name}: {state}')
    for partition in PARTITIONS:
        count = counts[partition]
        print(f'{partition:12s} {count:5d}/{EXPECTED_PER_PARTITION} '
              f'({100 * count / EXPECTED_PER_PARTITION:5.1f}%)')
    total = sum(counts.values())
    expected = EXPECTED_PER_PARTITION * len(PARTITIONS)
    print(f'total        {total:5d}/{expected} ({100 * total / expected:.1f}%)')


if __name__ == '__main__':
    main()
