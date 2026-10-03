"""Compare matched evaluation metrics from two dynamics checkpoints."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as source:
        return list(csv.DictReader(source))


def indexed(path):
    return {(row["task"], int(row["horizon"])): row for row in read_rows(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--new-metrics", type=Path, required=True)
    parser.add_argument("--old-normalized", type=Path, required=True)
    parser.add_argument("--new-normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_metrics = indexed(args.old_metrics)
    new_metrics = indexed(args.new_metrics)
    old_normalized = indexed(args.old_normalized)
    new_normalized = indexed(args.new_normalized)
    keys = old_metrics.keys()
    if not (keys == new_metrics.keys() == old_normalized.keys() == new_normalized.keys()):
        raise ValueError("all reports must contain the same task/horizon groups")

    rows = []
    for task, horizon in sorted(keys):
        old_m, new_m = old_metrics[(task, horizon)], new_metrics[(task, horizon)]
        old_n, new_n = old_normalized[(task, horizon)], new_normalized[(task, horizon)]
        old_scale = float(old_n["task_state_latent_std"])
        new_scale = float(new_n["task_state_latent_std"])
        if abs(old_scale - new_scale) > 1e-9:
            raise ValueError(f"{task}: task-state scales differ")
        old_latent = float(old_m["terminal_latent_rms"])
        new_latent = float(new_m["terminal_latent_rms"])
        old_image = float(old_m["terminal_image_rms"])
        new_image = float(new_m["terminal_image_rms"])
        rows.append({
            "task": task,
            "domain": "ManiSkill" if task.startswith("ms-") else "MetaWorld",
            "horizon": horizon,
            "old_latent_rms": old_latent,
            "finetuned_latent_rms": new_latent,
            "latent_change_pct": 100 * (new_latent / old_latent - 1),
            "old_image_rms": old_image,
            "finetuned_image_rms": new_image,
            "image_change_pct": 100 * (new_image / old_image - 1),
            "old_state_normalized": float(old_n["error_over_task_state_std"]),
            "finetuned_state_normalized": float(new_n["error_over_task_state_std"]),
            "old_copy_normalized": float(old_n["normalized_latent_error"]),
            "finetuned_copy_normalized": float(new_n["normalized_latent_error"]),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["horizon"])].append(row)
    means = []
    for (domain, horizon), items in sorted(grouped.items()):
        means.append({
            "domain": domain,
            "horizon": horizon,
            "tasks": len(items),
            **{field: statistics.mean(row[field] for row in items)
               for field in (
                   "old_latent_rms", "finetuned_latent_rms",
                   "old_image_rms", "finetuned_image_rms",
                   "old_state_normalized", "finetuned_state_normalized",
                   "old_copy_normalized", "finetuned_copy_normalized",
               )},
        })
    with (args.output / "domain_means.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ManiSkill": "#2563eb", "MetaWorld": "#ea580c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = (
        ("state_normalized", "Latent RMS / task-state SD", "State-normalized latent error"),
        ("latent_rms", "Latent RMS", "Raw latent error"),
        ("image_rms", "Image RMS", "Decoded-image error"),
    )
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for domain, color in colors.items():
            items = sorted((row for row in means if row["domain"] == domain),
                           key=lambda row: row["horizon"])
            for prefix, linestyle, marker, label in (
                ("old", "--", "x", "published"),
                ("finetuned", "-", "o", "fine-tuned"),
            ):
                ax.plot([row["horizon"] for row in items],
                        [row[f"{prefix}_{metric}"] for row in items],
                        color=color, linestyle=linestyle, marker=marker,
                        label=f"{domain} · {label}")
        ax.set(title=title, xlabel="Rollout horizon", ylabel=ylabel,
               xticks=[1, 4, 8, 16])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    task_count = len({row["task"] for row in rows})
    fig.suptitle(f"Published vs fine-tuned dynamics checkpoint ({task_count} matched tasks)",
                 fontsize=14, fontweight="bold")
    fig.savefig(args.output / "domain_comparison.png", dpi=180)
    plt.close(fig)

    longest = max(row["horizon"] for row in rows)
    task_rows = sorted((row for row in rows if row["horizon"] == longest),
                       key=lambda row: (row["domain"], row["latent_change_pct"]))
    fig, ax = plt.subplots(figsize=(10, 14), constrained_layout=True)
    values = [row["latent_change_pct"] for row in task_rows]
    bar_colors = ["#16a34a" if value < 0 else "#dc2626" for value in values]
    y = list(range(len(task_rows)))
    ax.barh(y, values, color=bar_colors, alpha=.85)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set(yticks=y, yticklabels=[row["task"] for row in task_rows],
           xlabel="Fine-tuned latent RMS change vs published checkpoint (%)",
           title=f"Per-task latent-error change at horizon {longest}\n"
                 "green = improvement; red = regression")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.savefig(args.output / "h16_task_deltas.png", dpi=180)
    plt.close(fig)

    # Rank by the fine-tuned checkpoint's shared-scale normalized error while
    # retaining the regular value beside it for a direct checkpoint comparison.
    ranked = sorted((row for row in rows if row["horizon"] == longest),
                    key=lambda row: row["finetuned_state_normalized"], reverse=True)
    y = list(range(len(ranked)))
    height = .38
    fig, ax = plt.subplots(figsize=(11, 14), constrained_layout=True)
    ax.barh([position - height / 2 for position in y],
            [row["old_state_normalized"] for row in ranked],
            height=height, color="#64748b", label="Published")
    ax.barh([position + height / 2 for position in y],
            [row["finetuned_state_normalized"] for row in ranked],
            height=height, color="#7c3aed", label="Fine-tuned")
    ax.set(yticks=y,
           yticklabels=[f"{rank}. {row['task']}" for rank, row in enumerate(ranked, 1)],
           xlabel="Latent RMS / task-state SD",
           title=f"Task error ranking at horizon {longest}\n"
                 "ranked by fine-tuned normalized error; highest error first")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.legend()
    fig.savefig(args.output / "h16_error_ranking.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {len(rows)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
