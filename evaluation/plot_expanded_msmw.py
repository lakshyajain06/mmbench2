"""Plot domain means for a new MS/MW cohort and the pooled validation tasks."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(summary_path, audit_path):
    rows = list(csv.DictReader(Path(summary_path).open(newline='')))
    audit = json.loads(Path(audit_path).read_text())
    selected = audit['tasks']
    new = (set(selected['maniskill'] + selected['metaworld'])
           if isinstance(selected, dict)
           else {item['task'] for item in selected})
    all_tasks = {row['task'] for row in rows}
    if not new <= all_tasks:
        raise ValueError(f'audited tasks missing from summary: {sorted(new - all_tasks)}')
    previous = all_tasks - new
    result = []
    for cohort, allowed in (('previous', previous), ('new', new), ('all', all_tasks)):
        for domain, prefix in (('ManiSkill', 'ms-'), ('MetaWorld', 'mw-')):
            for horizon in sorted({int(row['horizon']) for row in rows}):
                group = [row for row in rows if row['partition'] == 'val'
                         and row['task'] in allowed and row['task'].startswith(prefix)
                         and int(row['horizon']) == horizon]
                if not group:
                    raise ValueError(f'missing {cohort}/{domain}/{horizon}')
                result.append(dict(cohort=cohort, domain=domain, horizon=horizon,
                                   tasks=len(group), windows=sum(int(row['windows']) for row in group),
                                   latent_rms=statistics.mean(float(row['model_latent_rms']) for row in group),
                                   state_normalized=statistics.mean(float(row['error_over_task_state_std'])
                                                                    for row in group),
                                   copy_normalized=statistics.mean(float(row['normalized_latent_error'])
                                                                   for row in group)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.summary, args.audit)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'domain_means.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    counts = {cohort: sum(row['tasks'] for row in rows
                          if row['cohort'] == cohort and row['horizon'] == 1)
              for cohort in ('previous', 'new', 'all')}
    fig.suptitle(f'Validation: {counts["previous"]} previous + {counts["new"]} new tasks')
    colors = {'ManiSkill': '#2563eb', 'MetaWorld': '#ea580c'}
    for ax, metric, ylabel in zip(axes,
                                  ('state_normalized', 'latent_rms'),
                                  ('Latent RMS / task-state SD', 'Raw latent RMS')):
        for cohort, linestyle in (('previous', ':'), ('new', '-'), ('all', '--')):
            for domain in colors:
                group = sorted((row for row in rows if row['cohort'] == cohort
                                and row['domain'] == domain), key=lambda row: row['horizon'])
                ax.plot([row['horizon'] for row in group], [row[metric] for row in group],
                        color=colors[domain], linestyle=linestyle, marker='o',
                        label=f'{domain} · {cohort}')
        ax.set(xlabel='Rollout horizon (transitions)', ylabel=ylabel,
               xticks=[1, 4, 8, 16])
        ax.grid(alpha=.2)
    axes[0].set_title('Task-scale-normalized error')
    axes[1].set_title('Unnormalized error')
    axes[0].legend(fontsize=8)
    fig.savefig(args.output / 'domain_comparison.png', dpi=160)
    plt.close(fig)
    print(f"Wrote {len(rows)} domain means and plot to {args.output}")


if __name__ == '__main__':
    main()
