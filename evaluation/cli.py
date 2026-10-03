"""CLI for rollout, review, selection and confirmation."""
import argparse
import json
from pathlib import Path

from config import read_config
from report import episode_summary, make_review_sheet, select_tasks, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/robotics.json'))
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON_VALUE')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--partitions', nargs='+')
    prepare.add_argument('--tasks', nargs='+')
    prepare.add_argument('--workers', type=int, default=4)
    run = sub.add_parser('run')
    run.add_argument('--partitions', nargs='+')
    run.add_argument('--tasks', nargs='+')
    run.add_argument('--pilot', action='store_true')
    run.add_argument('--metrics-only-test', action='store_true',
                     help='Run raw metrics on test candidates before val selection; confirmation stays locked')
    sub.add_parser('review-sheet')
    diversity = sub.add_parser('diversity')
    diversity.add_argument('--partition', default='val')
    report = sub.add_parser('report')
    report.add_argument('--labels', required=True)
    report.add_argument('--partition')
    select = sub.add_parser('select')
    select.add_argument('--labels', required=True)
    calibrate = sub.add_parser('calibrate')
    calibrate.add_argument('--labels', required=True)
    confirm = sub.add_parser('confirm')
    confirm.add_argument('--labels', required=True)
    args = parser.parse_args()
    config = read_config(args.config, args.set)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        import subprocess
        import sys
        script = Path(__file__).resolve().parents[1] / 'src/preprocess_dataset.py'
        partitions = args.partitions or config['dataset']['partitions']
        tasks = args.tasks or config['tasks']['first_pass']
        for partition in partitions:
            command = [sys.executable, str(script), '--filedir',
                       str(Path(config['dataset']['root']) / partition), '--outdir',
                       str(Path(config['dataset']['shards_root']) / partition),
                       '--target_size', '224', '--num_workers', str(args.workers),
                       '--tasks', *tasks]
            subprocess.run(command, check=True, cwd=script.parent)
    elif args.command == 'run':
        partitions = args.partitions or config['dataset']['partitions']
        tasks = args.tasks or config['tasks']['first_pass']
        if 'test' in partitions and not args.metrics_only_test:
            lock = out / 'selection.json'
            if not lock.exists():
                raise SystemExit('test partition requires a saved val selection: run select first')
            selected = json.loads(lock.read_text())['tasks']
            if args.tasks is None:
                tasks = selected
            if tasks != selected:
                raise SystemExit(f'test tasks must match locked val selection: {selected}')
        from runner import run as run_rollouts
        print(run_rollouts(config, partitions, tasks, args.pilot,
                           exploratory_test=args.metrics_only_test))
    elif args.command == 'review-sheet':
        sheet = out / 'review.csv'
        print(f'{make_review_sheet(out / "windows.jsonl", sheet)} windows -> {sheet}')
    elif args.command == 'diversity':
        from diversity import describe
        lock = json.loads((out / 'selection.json').read_text())
        result = describe(out / 'steps.jsonl', lock['tasks'], args.partition)
        path = out / f'diversity_{args.partition}.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    elif args.command == 'calibrate':
        from predictors import calibrate as fit, onset_examples
        lock = json.loads((out / 'selection.json').read_text())
        examples = onset_examples(out / 'steps.jsonl', args.labels, 'val', lock['tasks'])
        result = fit(examples)
        (out / 'predictor_calibration.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    else:
        partition = 'val' if args.command == 'select' else 'test' if args.command == 'confirm' else args.partition
        rows, summary = summarize(out / 'steps.jsonl', args.labels, config['review'], partition)
        (out / 'window_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (out / 'episode_summary.json').write_text(json.dumps(episode_summary(rows), indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        if args.command == 'select':
            selected = select_tasks(summary)
            if not summary or any(row['reviewed'] != row['windows'] for row in summary):
                raise SystemExit('all val windows require labels before selection')
            if len(selected) < config['review']['min_selected_tasks']:
                raise SystemExit(f'only {len(selected)} tasks qualify; screen backups on diagnostic and val rounds')
            if len(selected) > config['review']['max_selected_tasks']:
                raise SystemExit(f'{len(selected)} tasks qualify; narrow the selected set on val before locking')
            lock = {'tasks': selected, 'thresholds': config['review'],
                    'source_partition': 'val', 'summary': summary}
            (out / 'selection.json').write_text(json.dumps(lock, indent=2) + '\n')
            print(f'locked {len(selected)} val tasks: {selected}')
        elif args.command == 'confirm':
            lock = json.loads((out / 'selection.json').read_text())
            if config['review'] != lock['thresholds']:
                raise SystemExit('test thresholds differ from locked val thresholds')
            if not summary or any(row['reviewed'] != row['windows'] for row in summary):
                raise SystemExit('all test windows require labels before confirmation')
            if sorted({r['task'] for r in summary}) != lock['tasks']:
                raise SystemExit('test task set differs from locked val selection')
            result = {'selected_on_val': lock['tasks'], 'test_summary': summary}
            calibration_path = out / 'predictor_calibration.json'
            if calibration_path.exists():
                from predictors import evaluate, onset_examples
                examples = onset_examples(out / 'steps.jsonl', args.labels, 'test', lock['tasks'])
                result['test_predictors'] = evaluate(examples, json.loads(calibration_path.read_text()))
            (out / 'confirmation.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
