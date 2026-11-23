"""Run the configurations already specified in report/task2_q_learning.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SETTINGS = {
    'default': [],
    'constant_alpha': ['--alpha', '0.1', '--alpha-decay', '0'],
    'constant_epsilon': ['--epsilon', '0.2', '--epsilon-decay', '0', '--epsilon-min', '0'],
    'optimistic': ['--initial-q', '5'],
}


def run_training(job):
    name, seed, args = job
    folder = args.output / name / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'train_q_learning.py'), '--output', str(folder),
               '--log-every', '10000', '--checkpoint-every', '100000']
    checkpoints = sorted(folder.glob('checkpoint_*.npz'))
    completed = 0
    if checkpoints:
        latest = checkpoints[-1]
        completed = int(latest.stem.split('_')[-1])
        command += ['--resume', str(latest)]
    else:
        command += ['--seed', str(seed)] + SETTINGS[name]
    if completed >= args.episodes:
        return f'{name}, seed {seed}: already finished'
    command += ['--episodes', str(args.episodes - completed)]
    with (folder / 'training.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((folder / 'results.json').read_text())
    return f"{name}, seed {seed}: {result['episodes']} episodes, {result['runtime_s']:.1f} training seconds"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/task2-final')
    parser.add_argument('--episodes', type=int, default=5_000_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--configurations', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        parser.error('episodes and workers must be positive')
    args.output = args.output.resolve()
    jobs = []
    for name in args.configurations:
        for seed in args.seeds:
            jobs.append((name, seed, args))
    print(f'Starting {len(jobs)} runs using {args.workers} workers. Logs: {args.output}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(run_training, jobs):
            print(message, flush=True)


if __name__ == '__main__':
    main()
