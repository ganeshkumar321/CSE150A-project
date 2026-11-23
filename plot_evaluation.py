"""Task 5: simple comparison figures and CSV tables from saved evaluation files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation_results import comparison_tables, write_csv
import poker_env as pe


def comparisons(source, output):
    """Build tables once, then pass them to the two comparison plots."""
    summaries, class_rows, labels = comparison_tables(source)
    write_csv(output / 'policy_comparison.csv', summaries)
    write_csv(output / 'by_class.csv', class_rows)
    plot_overall(summaries, output)
    plot_by_class(summaries, class_rows, labels, output)
    return summaries


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_overall(summaries, output):
    names = [row['policy'] for row in summaries]
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * .45)))
    for i, row in enumerate(summaries):
        error = 0
        if row['ci95_high'] is not None:
            error = row['ci95_high'] - row['mean']
        ax.errorbar(row['mean'], i, xerr=error, fmt='o', capsize=3)
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel='Mean payout (9/6 Jacks or Better)',
           title='Held-out Monte Carlo evaluation: approximate 95% CI')
    ax.grid(axis='x', alpha=.2)
    save_figure(fig, output / 'policy_comparison.png')


def plot_by_class(summaries, class_rows, labels, output):
    names = [row['policy'] for row in summaries]
    present = []
    for c in range(len(pe.UCI_CLASS_NAMES)):
        if np.any(labels == c):
            present.append(c)
    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * .45)))
    scores = {}
    for row in class_rows:
        scores[row['policy'], row['starting_class']] = row['mean']
    matrix = []
    for name in names:
        matrix.append([scores[name, pe.UCI_CLASS_NAMES[c]] for c in present])
    image = ax.imshow(matrix, aspect='auto', cmap='Blues')
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if value > np.max(matrix) * .6 else 'black')
    ticks = [f"{pe.UCI_CLASS_NAMES[c].replace('_', ' ')}\n(n={sum(labels == c):,})" for c in present]
    ax.set(xticks=range(len(present)), xticklabels=ticks, yticks=range(len(names)), yticklabels=names,
           title='Mean payout by starting hand class (9/6)')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    fig.colorbar(image, ax=ax, label='Payout')
    save_figure(fig, output / 'by_class.png')


def learning_curves(paths, output):
    """Plot existing Task 4 exact-score records, without treating them as MC draws."""
    groups = {}
    sources = []
    for path in paths:
        run = json.loads(path.read_text())
        if run['config'].get('reward') != 'job_9_6':
            continue
        sources.append(str(path))
        name = run['config']['abstraction']
        if name not in groups:
            groups[name] = []
        groups[name].append(run)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for name, runs in sorted(groups.items()):
        if len({r['seed'] for r in runs}) != len(runs):
            raise ValueError('duplicate Task 4 training seed in one abstraction')
        grids = []
        scores = []
        for run in runs:
            episodes = []
            payouts = []
            for point in run['curve']:
                episodes.append(point['episodes'])
                payouts.append(point['eval_mean'])
            if grids and episodes != grids[0]:
                raise ValueError('learning curves must use the same checkpoint episodes')
            grids.append(episodes)
            scores.append(payouts)
        y = np.array(scores)
        mean = y.mean(axis=0)
        sd = np.zeros(len(mean))
        if len(runs) > 1:
            sd = y.std(axis=0, ddof=1)
        line, = ax.plot(grids[0], mean, label=f'{name} ({len(runs)} seeds)')
        if len(runs) > 1:
            ax.fill_between(grids[0], mean-sd, mean+sd, color=line.get_color(), alpha=.15)
        for i, episodes in enumerate(grids[0]):
            rows.append(dict(abstraction=name, episodes=episodes, mean=float(mean[i]),
                             training_seed_sd=float(sd[i]) if len(runs)>1 else None,
                             n_training_seeds=len(runs),
                             method='Task 4 saved exact expectation on eval hands'))
    ax.set(xscale='log', xlabel='Training episodes', ylabel='Expected payout on eval hands (9/6)',
           title='Saved Task 4 runs: mean and ±1 SD across training seeds')
    ax.legend()
    ax.grid(alpha=.2)
    save_figure(fig, output / 'learning_curves.png')
    write_csv(output / 'learning_curves.csv', rows)
    seed_rows = []
    for name, runs in sorted(groups.items()):
        for run in runs:
            seed_rows.append(dict(abstraction=name, seed=run['seed'], episodes=run['episodes'],
                                  eval_mean=run['eval_mean'], training_runtime_s=run['runtime_s'],
                                  method='Task 4 saved exact expectation'))
    write_csv(output / 'training_seed_comparison.csv', seed_rows)
    (output / 'learning_curve_sources.json').write_text(json.dumps(sources, indent=2)+'\n')


def checkpoint_curves(source, output):
    """Task 2 curves: average evaluation draws first, then compare training seeds."""
    data = json.loads(source.read_text())
    groups = {}
    for record in data['records']:
        if 'checkpoint' not in record:
            continue
        # Group by training settings; the seed varies within each configuration.
        config = record['config'].copy()
        config.pop('seed', None)
        config['training_paytable'] = record.get('metadata', {}).get('paytable')
        group = json.dumps(config, sort_keys=True)
        if group not in groups:
            groups[group] = {}
        seed = record['seed']
        if seed not in groups[group]:
            groups[group][seed] = {}
        episode = record['episodes']
        if episode not in groups[group][seed]:
            groups[group][seed][episode] = []
        groups[group][seed][episode].append(record['eval_mean'])
    if not groups:
        return
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, group in enumerate(sorted(groups)):
        for seed, scores in sorted(groups[group].items()):
            episodes = sorted(scores)
            values = [float(np.mean(scores[e])) for e in episodes]
            ax.plot(episodes, values, marker='.', label=f'config {index+1}, train seed {seed}')
            for e,v in zip(episodes,values):
                rows.append(dict(configuration=index+1, config=group, training_seed=seed,
                                 episodes=e, eval_mean=v,
                                 n_eval_seeds=len(scores[e])))
    ax.set(xlabel='Training episodes', ylabel='Mean payout (9/6)',
           title='Saved Task 2 checkpoints: one curve per training seed')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    save_figure(fig, output / 'checkpoint_learning_curves.png')
    write_csv(output / 'checkpoint_learning_curves.csv', rows)


def greedy_figures(source, output):
    """Summarize Task 2 exact scores separately from the Monte Carlo baselines."""
    if not source.exists():
        return
    data = json.loads(source.read_text())
    if not data['complete']:
        return
    groups = {}
    for record in data['records']:
        name = record['configuration']
        if name not in groups:
            groups[name] = {}
        episode = record['episodes']
        if episode not in groups[name]:
            groups[name][episode] = []
        groups[name][episode].append(record)
    curve_rows = []
    final_rows = []
    class_rows = []
    fig, ax = plt.subplots(figsize=(7,4))
    for name, checkpoints in sorted(groups.items()):
        episodes, means, deviations = [], [], []
        for episode, runs in sorted(checkpoints.items()):
            if len(runs) != 5 or len({r['seed'] for r in runs}) != 5:
                raise ValueError('each checkpoint needs all five training seeds')
            scores = [r['eval_mean'] for r in runs]
            mean = float(np.mean(scores))
            sd = float(np.std(scores,ddof=1))
            episodes.append(episode)
            means.append(mean)
            deviations.append(sd)
            curve_rows.append(dict(configuration=name,episodes=episode,mean=mean,
                                   training_seed_sd=sd,n_training_seeds=len(runs)))
            if episode == data['target_episodes']:
                final_rows.append(dict(configuration=name,mean=mean,training_seed_sd=sd,
                    n_training_seeds=len(runs),episodes=episode,
                    visited_action_fraction=float(np.mean([r['visited_action_fraction'] for r in runs]))))
                for cls in pe.UCI_CLASS_NAMES:
                    values = [r['by_class'][cls] for r in runs if cls in r['by_class']]
                    value = float(np.mean(values)) if values else None
                    class_rows.append(dict(configuration=name,starting_class=cls,mean=value))
        means, deviations = np.array(means), np.array(deviations)
        line, = ax.plot(episodes,means,marker='.',label=name.replace('_',' '))
        ax.fill_between(episodes,means-deviations,means+deviations,color=line.get_color(),alpha=.15)
    ax.set(xlabel='Training episodes',ylabel='Expected payout on eval hands (9/6)',
           title='Task 2 greedy policies: mean and ±1 SD across training seeds')
    ax.ticklabel_format(axis='x',style='sci',scilimits=(6,6))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    save_figure(fig,output/'task2_learning_curves.png')
    write_csv(output/'task2_learning_curves.csv',curve_rows)
    write_csv(output/'task2_final_comparison.csv',final_rows)
    write_csv(output/'task2_by_class.csv',class_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=Path('results/task5/evaluation.json'))
    parser.add_argument('--output', type=Path, default=Path('report/figures/task5'))
    parser.add_argument('--task4-runs', type=Path, default=Path('results/task4/runs'))
    parser.add_argument('--greedy-results', type=Path, default=Path('results/task5-greedy/checkpoints.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = comparisons(args.evaluation, args.output)
    checkpoint_curves(args.evaluation, args.output)
    learning_curves(sorted(args.task4_runs.glob('*.json')), args.output)
    greedy_figures(args.greedy_results,args.output)
    for row in summaries:
        print(f"{row['policy']}: {row['mean']:.6f} [{row['ci95_low']}, {row['ci95_high']}]")


if __name__ == '__main__':
    main()
