#!/usr/bin/env python3
"""Plot and summarize matched-resolution PPO evaluation logs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(args.logs):
        records = read_jsonl(path)
        if not records:
            continue
        config_path = path.parent / "config.json"
        if not config_path.exists():
            config_path = path.parents[1] / "config.json"
        config = json.loads(config_path.read_text())
        task = str(config["task"])
        runs.append((task, path, records))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for task, _path, records in runs:
        steps = [int(record["timesteps"]) for record in records]
        axes[0].plot(
            steps,
            [float(record["success_rate"]) for record in records],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=task,
        )
        axes[1].plot(
            steps,
            [float(record["mean_return"]) for record in records],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=task,
        )
    axes[0].set_title("Evaluation success during PPO training")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_title("Evaluation return during PPO training")
    axes[1].set_ylabel("Mean return")
    for axis in axes:
        axis.set_xlabel("Environment transitions")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Matched 224×224 visual PPO training results", fontsize=15)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(args.output_dir / "matched224_training_curves.png", dpi=180)
    plt.close(fig)

    summary = []
    for task, path, records in runs:
        best = max(
            records,
            key=lambda record: (
                float(record["success_rate"]), float(record["mean_return"])
            ),
        )
        latest = records[-1]
        summary.append(
            {
                "task": task,
                "evaluations": len(records),
                "latest_timesteps": int(latest["timesteps"]),
                "latest_success_rate": float(latest["success_rate"]),
                "latest_mean_return": float(latest["mean_return"]),
                "best_success_rate": float(best["success_rate"]),
                "best_success_timesteps": int(best["timesteps"]),
                "return_at_best_success": float(best["mean_return"]),
                "source": str(path),
            }
        )
    fields = list(summary[0]) if summary else []
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "snapshot_utc": datetime.now(timezone.utc).isoformat(),
                "runs": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
