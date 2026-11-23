"""Task 5: exact greedy checkpoint scores using Task 4's existing evaluator.

python evaluate_checkpoints.py --watch
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from evaluate_policies import load_checkpoint
from evaluation_results import file_hash, write_csv
from poker_data import load_split, SPLIT_DIR
from task4_experiments import score

CONFIGURATIONS = ['default', 'constant_alpha', 'constant_epsilon', 'optimistic']


def score_checkpoint(path, split):
    agent = load_checkpoint(path)
    if agent.state_encoding != 'canonical':
        raise ValueError('This exact-score summary expects Task 2 canonical checkpoints')
    # These actions are already in canonical positions, as required by Task 4.
    result = score(np.argmax(agent.Q, axis=1), split)
    result.update(checkpoint=str(path), checkpoint_sha256=file_hash(path),
                  configuration=path.parent.parent.name, seed=agent.seed, episodes=agent.episode,
                  config=agent.get_config(), training_runtime_s=agent.runtime_s,
                  visited_state_fraction=float(np.any(agent.N > 0, axis=1).mean()),
                  visited_action_fraction=float((agent.N > 0).mean()),
                  method='exact expected payout on shared eval hands; greedy policy')
    return result


def save_summary(records, output, target, expected_runs):
    finals = []
    for name in CONFIGURATIONS:
        runs = []
        for record in records:
            if record['configuration'] == name and record['episodes'] == target:
                runs.append(record)
        if not runs:
            continue
        means = [r['eval_mean'] for r in runs]
        sd = float(np.std(means, ddof=1)) if len(runs) > 1 else None
        finals.append(dict(configuration=name, episodes=target, n_training_seeds=len(runs),
                           mean=float(np.mean(means)), training_seed_sd=sd,
                           mean_training_runtime_s=float(np.mean([r['training_runtime_s'] for r in runs])),
                           visited_state_fraction=float(np.mean([r['visited_state_fraction'] for r in runs])),
                           visited_action_fraction=float(np.mean([r['visited_action_fraction'] for r in runs]))))
    output.mkdir(parents=True, exist_ok=True)
    complete = sum(row['n_training_seeds'] for row in finals) == expected_runs
    data = dict(method='exact', complete=complete, target_episodes=target,
                split_sha256=file_hash(Path(SPLIT_DIR)/'eval.npz'), records=records)
    (output/'checkpoints.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    write_csv(output/'final_comparison.csv', finals)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/task2-final'))
    parser.add_argument('--output', type=Path, default=Path('results/task5-greedy'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    split = load_split('eval')
    records = []
    saved = args.output/'checkpoints.json'
    if saved.exists():
        data = json.loads(saved.read_text())
        if data['split_sha256'] != file_hash(Path(SPLIT_DIR)/'eval.npz'):
            raise ValueError('Saved scores use a different evaluation split')
        records = data['records']
        for record in records:
            if record['checkpoint_sha256'] != file_hash(record['checkpoint']):
                raise ValueError('a previously scored checkpoint has changed')
    seen = {r['checkpoint'] for r in records}
    while True:
        for name in CONFIGURATIONS:
            for seed in range(5):
                # A few matched checkpoints are enough for a readable learning curve.
                for episode in [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]:
                    path = args.runs/name/f'seed{seed}'/f'checkpoint_{episode:09d}.npz'
                    if path.exists() and str(path) not in seen:
                        result = score_checkpoint(path, split)
                        records.append(result)
                        seen.add(str(path))
                        print(f"{name}, seed {seed}, {episode}: {result['eval_mean']:.6f}", flush=True)
        complete = save_summary(records,args.output,5_000_000,20)
        if complete or not args.watch:
            if complete:
                from plot_evaluation import greedy_figures
                figures = Path('report/figures/task5')
                figures.mkdir(parents=True,exist_ok=True)
                greedy_figures(saved,figures)
            print(f'Final evaluation complete: {complete}',flush=True)
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
