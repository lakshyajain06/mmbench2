"""Plot per-task latent error curves from a normalized evaluation summary."""

import argparse
import csv
from collections import defaultdict
from html import escape
from pathlib import Path


HORIZONS = (1, 4, 8, 16)


def load_tasks(path):
    tasks = defaultdict(dict)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["partition"] != "val":
                raise ValueError(f"expected validation-only summary: {path}")
            task, horizon = row["task"], int(row["horizon"])
            if horizon in tasks[task]:
                raise ValueError(f"duplicate task/horizon: {task}/{horizon}")
            tasks[task][horizon] = {
                "raw": float(row["model_latent_rms"]),
                "state": float(row["error_over_task_state_std"]),
                "copy": float(row["normalized_latent_error"]),
                "windows": int(row["windows"]),
            }
    if not tasks:
        raise ValueError("empty summary")
    for task, values in tasks.items():
        if set(values) != set(HORIZONS):
            raise ValueError(f"incomplete horizons for {task}: {sorted(values)}")
    return dict(sorted(tasks.items()))


def plot_task(task, values, output, limits):
    import matplotlib.pyplot as plt

    color = "#2563eb" if task.startswith("ms-") else "#ea580c"
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    for ax, key, title, ylabel, limit in zip(
        axes,
        ("state", "raw", "copy"),
        ("Task-state normalized", "Raw latent error", "Copy-start normalized"),
        ("Model RMS / real-state SD", "Latent RMS", "Model RMS / copy-start RMS"),
        limits,
    ):
        series = [values[h][key] for h in HORIZONS]
        ax.plot(HORIZONS, series, color=color, marker="o", linewidth=2)
        ax.set(xlabel="Prediction steps", ylabel=ylabel, title=title,
               xticks=HORIZONS, xlim=(0, 17), ylim=(0, limit))
        ax.grid(alpha=0.2)
        if key == "copy":
            ax.axhline(1, color="#475569", linestyle="--", linewidth=1)
        ax.annotate(f"{series[-1]:.3f}", (16, series[-1]), xytext=(-4, 8),
                    textcoords="offset points", ha="right", fontsize=9)
    fig.suptitle(f"{task} · validation · {values[16]['windows']} windows/horizon",
                 fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_overview(tasks, output):
    import matplotlib.pyplot as plt

    ordered = sorted(tasks, key=lambda task: (not task.startswith("ms-"),
                                               -tasks[task][16]["state"]))
    labels = [f"{task}  ({'MS' if task.startswith('ms-') else 'MW'})" for task in ordered]
    colors = ["#2563eb" if task.startswith("ms-") else "#ea580c" for task in ordered]
    y = list(range(len(ordered)))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(10, len(ordered) * 0.38 + 2)), sharey=True,
                             constrained_layout=True)
    for ax, key, title, xlabel in zip(
        axes, ("state", "raw"),
        ("Normalized latent error at 16 steps", "Raw latent error at 16 steps"),
        ("Model RMS / real-state SD", "Latent RMS"),
    ):
        values = [tasks[task][16][key] for task in ordered]
        ax.barh(y, values, color=colors, alpha=0.85)
        ax.set(xlabel=xlabel, title=title, yticks=y)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values) * 1.17)
        for pos, value in enumerate(values):
            ax.text(value + max(values) * 0.015, pos, f"{value:.3f}",
                    va="center", fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("MMBench2 validation · each task weighted equally · 40 windows/task/horizon",
                 fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_index(tasks, output, protocol):
    sections = []
    for name, prefix in (("ManiSkill", "ms-"), ("MetaWorld", "mw-")):
        cards = []
        for task in tasks:
            if not task.startswith(prefix):
                continue
            filename = f"tasks/{task}.png"
            cards.append(f'<a class="card" href="{filename}">'
                         f'<strong>{escape(task)}</strong>'
                         f'<span>16-step normalized: {tasks[task][16]["state"]:.3f}</span>'
                         f'<img loading="lazy" src="{filename}" alt="Error curves for {escape(task)}"></a>')
        sections.append(f"<h2>{name}</h2><div class=grid>{''.join(cards)}</div>")
    output.write_text("<!doctype html><html lang=en><meta charset=utf-8>"
                      "<meta name=viewport content='width=device-width,initial-scale=1'>"
                      "<title>MMBench2 validation per-task latent error</title>"
                      "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 18px;"
                      "color:#172033}p{line-height:1.5}.grid{display:grid;"
                      "grid-template-columns:repeat(auto-fit,minmax(490px,1fr));gap:16px}"
                      ".card{border:1px solid #d7dce5;border-radius:8px;padding:12px;"
                      "color:inherit;text-decoration:none}.card span{float:right}img{width:100%;"
                      "margin-top:8px}a:hover{border-color:#2563eb}</style>"
                      "<h1>MMBench2 validation: per-task latent error</h1>"
                      "<p>Each point averages 40 windows (20 episodes × 2 seeds). "
                      f"{escape(protocol)} "
                      "Task-state normalization divides prediction RMS by that task's "
                      "real-state latent standard deviation estimated from validation frames. "
                      "The dashed copy-start line is 1: below it the model beats copying "
                      "the starting latent. These are fidelity metrics, not physical "
                      "hallucination labels.</p>"
                      "<p><a href=overview.png>Open the 26-task overview</a></p>"
                      + "".join(sections) + "</html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", default="All tasks use the same 16%-of-episode start protocol.")
    args = parser.parse_args()
    tasks = load_tasks(args.summary)
    args.output.mkdir(parents=True, exist_ok=True)
    task_dir = args.output / "tasks"
    task_dir.mkdir(exist_ok=True)
    limits = tuple(max(values[h][key] for values in tasks.values() for h in HORIZONS) * 1.2
                   for key in ("state", "raw", "copy"))
    for task, values in tasks.items():
        plot_task(task, values, task_dir / f"{task}.png", limits)
    plot_overview(tasks, args.output / "overview.png")
    write_index(tasks, args.output / "index.html", args.protocol)
    print(f"Wrote {len(tasks)} task graphs and an overview to {args.output}")


if __name__ == "__main__":
    main()
