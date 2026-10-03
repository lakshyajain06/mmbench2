#!/usr/bin/env python3
"""Plot simulator and world-model returns from paired policy evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--csv", dest="csv_path", type=Path)
    return p


def policy_label(path: Path) -> str:
    name = path.parent.name.lower()
    if name.startswith("skrl"):
        return "Current skrl PPO"
    return path.parent.name.replace("_", " ")


def main() -> None:
    args = parser().parse_args()
    records = []
    for path in args.metrics:
        metrics = json.loads(path.read_text())
        records.append(
            {
                "policy": policy_label(path),
                "task": metrics["task"],
                "episodes": int(metrics["episodes"]),
                "steps_per_episode": int(metrics["steps_per_episode"]),
                "simulator_return_mean": float(metrics["simulator_return_mean"]),
                "simulator_return_std": float(metrics["simulator_return_std"]),
                "world_model_return_mean": float(metrics["predicted_return_mean"]),
                "world_model_return_std": float(metrics["predicted_return_std"]),
                "simulator_success_rate": float(metrics["simulator_success_rate"]),
                "world_model_success_rate": "unavailable",
            }
        )

    labels = [record["policy"] for record in records]
    x = np.arange(len(records))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    ax.bar(
        x - width / 2,
        [record["simulator_return_mean"] for record in records],
        width,
        yerr=[record["simulator_return_std"] for record in records],
        capsize=5,
        label="Simulator: true reward",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [record["world_model_return_mean"] for record in records],
        width,
        yerr=[record["world_model_return_std"] for record in records],
        capsize=5,
        label="World model: predicted reward",
        color="#F58518",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean 25-step episode return")
    ax.set_title("ms-reach policy evaluation: simulator vs combined_xl")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.02,
        "Bars show means across 20 paired seeds; error bars show episode standard deviation.\n"
        "World-model return is produced by the learned reward head; it is not a success rate.",
        fontsize=9,
        color="#444444",
        va="bottom",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
