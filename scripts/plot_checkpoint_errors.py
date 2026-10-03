#!/usr/bin/env python3
"""Merge sharded rollout metrics and produce checkpoint-comparison charts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"combined": "#6C7A89", "combined_xl": "#E45756"}


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    unique = {(row["checkpoint"], row["task"]): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row["domain"], row["task"], row["checkpoint"]))


def paired(rows: list[dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for row in rows:
        result.setdefault(row["task"], {})[row["checkpoint"]] = row
    return {task: values for task, values in result.items() if len(values) == 2}


def grouped_horizontal_chart(
    pairs: dict[str, dict[str, dict]],
    *,
    metric: str,
    xlabel: str,
    output: Path,
    reference: float | None = None,
) -> None:
    domains = [("maniskill", "ManiSkill"), ("metaworld", "MetaWorld")]
    counts = [sum(next(iter(v.values()))["domain"] == domain for v in pairs.values()) for domain, _ in domains]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, max(9, 0.34 * max(counts))),
        gridspec_kw={"width_ratios": [max(counts[0], 1), max(counts[1], 1)]},
        constrained_layout=True,
    )
    labels = ["combined", "combined_xl"]
    for axis, (domain, title) in zip(axes, domains):
        tasks = sorted(task for task, values in pairs.items() if next(iter(values.values()))["domain"] == domain)
        y = np.arange(len(tasks))
        height = 0.38
        for offset, label in zip((-height / 2, height / 2), labels):
            values = [pairs[task][label][metric] for task in tasks]
            axis.barh(y + offset, values, height=height, label=label, color=COLORS[label])
        axis.set_yticks(y, [task.removeprefix("ms-").removeprefix("mw-") for task in tasks], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
        if reference is not None:
            axis.axvline(reference, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def delta_chart(pairs: dict[str, dict[str, dict]], output: Path) -> None:
    domains = [("maniskill", "ManiSkill"), ("metaworld", "MetaWorld")]
    counts = [sum(next(iter(v.values()))["domain"] == domain for v in pairs.values()) for domain, _ in domains]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, max(9, 0.34 * max(counts))),
        gridspec_kw={"width_ratios": [max(counts[0], 1), max(counts[1], 1)]},
        constrained_layout=True,
    )
    for axis, (domain, title) in zip(axes, domains):
        tasks = sorted(task for task, values in pairs.items() if next(iter(values.values()))["domain"] == domain)
        delta = [
            100.0 * (pairs[task]["combined_xl"]["mse"] / pairs[task]["combined"]["mse"] - 1.0)
            for task in tasks
        ]
        colors = ["#2A9D8F" if value < 0 else "#E76F51" for value in delta]
        y = np.arange(len(tasks))
        axis.barh(y, delta, color=colors)
        axis.set_yticks(y, [task.removeprefix("ms-").removeprefix("mw-") for task in tasks], fontsize=8)
        axis.invert_yaxis()
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("combined_xl MSE change vs combined (%) — lower is better")
        axis.grid(axis="x", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ranked_task_chart(pairs: dict[str, dict[str, dict]], output: Path) -> None:
    """Rank tasks by XL normalized error and retain both checkpoint values."""
    tasks = sorted(
        pairs,
        key=lambda task: pairs[task]["combined_xl"]["mse_ratio_pred_over_floor"],
    )
    combined = np.asarray(
        [pairs[task]["combined"]["mse_ratio_pred_over_floor"] for task in tasks]
    )
    combined_xl = np.asarray(
        [pairs[task]["combined_xl"]["mse_ratio_pred_over_floor"] for task in tasks]
    )
    y = np.arange(len(tasks))
    display_names = [
        f"{rank + 1:02d}. {task}"
        for rank, task in enumerate(tasks)
    ]

    fig, axis = plt.subplots(figsize=(12, 0.32 * len(tasks) + 2), constrained_layout=True)
    for index in range(len(tasks)):
        domain_color = "#457B9D" if tasks[index].startswith("ms-") else "#2A9D8F"
        axis.plot(
            [combined[index], combined_xl[index]],
            [y[index], y[index]],
            color=domain_color,
            linewidth=1.5,
            alpha=0.45,
            zorder=1,
        )
    axis.scatter(combined, y, s=35, color=COLORS["combined"], label="combined", zorder=2)
    axis.scatter(combined_xl, y, s=40, color=COLORS["combined_xl"], label="combined_xl", zorder=3)
    axis.set_yticks(y, display_names, fontsize=8)
    axis.invert_yaxis()
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set_xlabel("Rollout MSE / last-frame baseline MSE — lower is better")
    axis.set_title("Task ranking by combined_xl normalized rollout error")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ranked_psnr_chart(pairs: dict[str, dict[str, dict]], output: Path) -> None:
    """Rank tasks by XL raw PSNR and retain both checkpoint values."""
    tasks = sorted(
        pairs,
        key=lambda task: pairs[task]["combined_xl"]["psnr"],
        reverse=True,
    )
    combined = np.asarray([pairs[task]["combined"]["psnr"] for task in tasks])
    combined_xl = np.asarray([pairs[task]["combined_xl"]["psnr"] for task in tasks])
    y = np.arange(len(tasks))
    display_names = [f"{rank + 1:02d}. {task}" for rank, task in enumerate(tasks)]

    fig, axis = plt.subplots(figsize=(12, 0.32 * len(tasks) + 2), constrained_layout=True)
    for index in range(len(tasks)):
        domain_color = "#457B9D" if tasks[index].startswith("ms-") else "#2A9D8F"
        axis.plot(
            [combined[index], combined_xl[index]],
            [y[index], y[index]],
            color=domain_color,
            linewidth=1.5,
            alpha=0.45,
            zorder=1,
        )
    axis.scatter(combined, y, s=35, color=COLORS["combined"], label="combined", zorder=2)
    axis.scatter(combined_xl, y, s=40, color=COLORS["combined_xl"], label="combined_xl", zorder=3)
    axis.set_yticks(y, display_names, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("16-step rollout PSNR (dB) — higher is better")
    axis.set_title("Task ranking by combined_xl rollout PSNR")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ranked_psnr_gain_chart(pairs: dict[str, dict[str, dict]], output: Path) -> None:
    """Rank tasks by XL PSNR gain over the last-frame baseline."""
    metric = "psnr_gain_over_floor_db"
    tasks = sorted(pairs, key=lambda task: pairs[task]["combined_xl"][metric], reverse=True)
    combined = np.asarray([pairs[task]["combined"][metric] for task in tasks])
    combined_xl = np.asarray([pairs[task]["combined_xl"][metric] for task in tasks])
    y = np.arange(len(tasks))
    display_names = [f"{rank + 1:02d}. {task}" for rank, task in enumerate(tasks)]

    fig, axis = plt.subplots(figsize=(12, 0.32 * len(tasks) + 2), constrained_layout=True)
    for index in range(len(tasks)):
        domain_color = "#457B9D" if tasks[index].startswith("ms-") else "#2A9D8F"
        axis.plot(
            [combined[index], combined_xl[index]],
            [y[index], y[index]],
            color=domain_color,
            linewidth=1.5,
            alpha=0.45,
            zorder=1,
        )
    axis.scatter(combined, y, s=35, color=COLORS["combined"], label="combined", zorder=2)
    axis.scatter(combined_xl, y, s=40, color=COLORS["combined_xl"], label="combined_xl", zorder=3)
    axis.set_yticks(y, display_names, fontsize=8)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set_xlabel("PSNR gain over last-frame baseline (dB) — higher is better")
    axis.set_title("Task ranking by combined_xl normalized PSNR gain")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ranked_psnr_gain_by_domain_chart(pairs: dict[str, dict[str, dict]], output: Path) -> None:
    """Rank PSNR gain separately by domain with a shared horizontal scale."""
    metric = "psnr_gain_over_floor_db"
    domains = [("maniskill", "ManiSkill"), ("metaworld", "MetaWorld")]
    all_values = [values[label][metric] for values in pairs.values() for label in ("combined", "combined_xl")]
    value_min = min(0.0, min(all_values))
    value_max = max(all_values)
    padding = max(0.25, 0.04 * (value_max - value_min))
    shared_limits = (value_min - padding, value_max + padding)

    domain_tasks = {
        domain: sorted(
            (
                task
                for task, values in pairs.items()
                if next(iter(values.values()))["domain"] == domain
            ),
            key=lambda task: pairs[task]["combined_xl"][metric],
            reverse=True,
        )
        for domain, _title in domains
    }
    max_count = max(len(tasks) for tasks in domain_tasks.values())
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19, 0.4 * max_count + 2.4),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (domain, title) in zip(axes, domains):
        tasks = domain_tasks[domain]
        combined = np.asarray([pairs[task]["combined"][metric] for task in tasks])
        combined_xl = np.asarray([pairs[task]["combined_xl"][metric] for task in tasks])
        y = np.arange(len(tasks))
        display_names = [f"{rank + 1:02d}. {task}" for rank, task in enumerate(tasks)]
        line_color = "#457B9D" if domain == "maniskill" else "#2A9D8F"
        for index in range(len(tasks)):
            axis.plot(
                [combined[index], combined_xl[index]],
                [y[index], y[index]],
                color=line_color,
                linewidth=1.5,
                alpha=0.45,
                zorder=1,
            )
        axis.scatter(combined, y, s=35, color=COLORS["combined"], label="combined", zorder=2)
        axis.scatter(combined_xl, y, s=40, color=COLORS["combined_xl"], label="combined_xl", zorder=3)
        axis.set_yticks(y, display_names, fontsize=8)
        axis.invert_yaxis()
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
        axis.set_xlim(*shared_limits)
        axis.set_title(title)
        axis.set_xlabel("PSNR gain over last-frame baseline (dB) — higher is better")
        axis.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Task ranking by combined_xl normalized PSNR gain", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.inputs)
    pairs = paired(rows)
    expected = {(row["checkpoint"], row["task"]) for row in rows}
    if len(expected) != len(rows):
        raise SystemExit("duplicate rows remain after merge")
    if not pairs:
        raise SystemExit("no paired task results found")

    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    scalar_fields = [
        "checkpoint", "task", "domain", "episodes", "context", "horizon", "mse",
        "mse_episode_std", "mse_episode_sem", "floor_mse", "mse_ratio_pred_over_floor",
        "psnr", "floor_psnr", "psnr_gain_over_floor_db", "seconds",
    ]
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)

    grouped_horizontal_chart(
        pairs,
        metric="mse_ratio_pred_over_floor",
        xlabel="Rollout MSE / last-frame baseline MSE — lower is better",
        output=args.output_dir / "per_task_normalized_mse.png",
        reference=1.0,
    )
    grouped_horizontal_chart(
        pairs,
        metric="mse",
        xlabel="16-step rollout pixel MSE — lower is better",
        output=args.output_dir / "per_task_raw_mse.png",
    )
    grouped_horizontal_chart(
        pairs,
        metric="psnr",
        xlabel="16-step rollout PSNR (dB) — higher is better",
        output=args.output_dir / "per_task_psnr.png",
    )
    delta_chart(pairs, args.output_dir / "per_task_mse_change.png")
    ranked_task_chart(pairs, args.output_dir / "ranked_tasks_normalized_mse.png")
    ranked_psnr_chart(pairs, args.output_dir / "ranked_tasks_psnr.png")
    ranked_psnr_gain_chart(pairs, args.output_dir / "ranked_tasks_psnr_gain.png")
    ranked_psnr_gain_by_domain_chart(
        pairs, args.output_dir / "ranked_tasks_psnr_gain_by_domain.png"
    )

    summary = {}
    for domain in ("maniskill", "metaworld", "all"):
        tasks = [
            task for task, values in pairs.items()
            if domain == "all" or next(iter(values.values()))["domain"] == domain
        ]
        summary[domain] = {
            label: {
                "tasks": len(tasks),
                "macro_mse": float(np.mean([pairs[task][label]["mse"] for task in tasks])),
                "macro_mse_ratio": float(np.mean([pairs[task][label]["mse_ratio_pred_over_floor"] for task in tasks])),
                "macro_psnr": float(np.mean([pairs[task][label]["psnr"] for task in tasks])),
                "macro_psnr_gain_db": float(np.mean([pairs[task][label]["psnr_gain_over_floor_db"] for task in tasks])),
            }
            for label in ("combined", "combined_xl")
        }
    summary["paired_tasks"] = len(pairs)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
