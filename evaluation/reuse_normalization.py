"""Apply reference run copy baselines and task scales to matched checkpoint errors."""
import argparse
import csv
import json
from pathlib import Path

from normalize_latent import IDENTITY, identity, read_reference_scales, summarize, write_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="normalized directory for the matched reference checkpoint")
    parser.add_argument("--run", type=Path, required=True,
                        help="aggregate raw run for the checkpoint being normalized")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = {}
    with (args.reference / "windows.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            reference[identity(row)] = row
    terminal = {}
    with (args.run / "steps.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["transition"] == row["horizon"]:
                terminal[identity(row)] = row
    if reference.keys() != terminal.keys():
        raise ValueError("checkpoint and reference window identities differ")

    rows = []
    for key in sorted(reference):
        base = reference[key]
        model = float(terminal[key]["latent_rms"])
        copy = float(base["copy_start_latent_rms"])
        rows.append(dict(**{field: base[field] for field in IDENTITY},
                         model_latent_rms=model,
                         copy_start_latent_rms=copy,
                         model_beats_copy=model < copy))
    scales = read_reference_scales(args.reference / "summary.csv")
    summary = summarize(rows, scales)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "windows.jsonl").open("w") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_figure(summary, args.output / "summary.png", "normalized_latent_error",
                 "Latent prediction error / real endpoint change",
                 "Model RMS / copy-start RMS")
    write_figure(summary, args.output / "state_normalized.png", "error_over_task_state_std",
                 "Latent prediction error / pooled real-state variation",
                 "Model RMS / task latent standard deviation")
    print(f"{len(rows)} matched windows; {len(summary)} groups; {args.output}")


if __name__ == "__main__":
    main()
