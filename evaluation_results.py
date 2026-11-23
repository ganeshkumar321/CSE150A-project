"""Task 5 summaries and result files, shared by evaluation and plotting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import poker_env as pe
from poker_data import load_split, SPLIT_DIR


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    """Mean and approximate 95% interval over starting-hand averages."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('summary values must be a finite one-dimensional array')
    count = len(values)
    if count == 0:
        return dict(n_hands=0, mean=None, se=None, ci95=None)
    mean = float(values.mean())
    se = None
    ci = None
    if count > 1:
        # Standard error uses the number of starting hands, not total draws.
        se = float(values.std(ddof=1) / np.sqrt(count))
        ci = [mean - 1.96 * se, mean + 1.96 * se]
    return dict(n_hands=count, mean=mean, se=se, ci95=ci)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_tables(source):
    """Average evaluation seeds per hand, then summarize overall and by class."""
    data = json.loads(source.read_text())
    if data['split_sha256'] != file_hash(Path(SPLIT_DIR) / 'eval.npz'):
        raise ValueError('evaluation file belongs to a different split')
    labels = load_split('eval')['uci_class']
    summaries, class_rows = [], []
    groups = {}
    for record in data['records']:
        name = record['policy']
        if name not in groups:
            groups[name] = []
        groups[name].append(record)
    with np.load(source.with_name('per_hand.npz')) as values:
        labels = labels[values['indices']]
        for name, runs in groups.items():
            seeds = []
            hand_results = []
            seed_means = []
            runtimes = []
            decision_times = []
            for run in runs:
                if run['eval_seed'] in seeds:
                    raise ValueError('repeated evaluation seed for one policy')
                seeds.append(run['eval_seed'])
                hand_results.append(values[run['values_key']])
                seed_means.append(run['eval_mean'])
                runtimes.append(run['runtime_s'])
                decision_times.append(run['decision_runtime_s'])

            # Average the seeds for each hand before calculating the interval.
            per_hand = np.mean(hand_results, axis=0)
            stats = summarize(per_hand)
            interval = stats['ci95'] or [None, None]
            seed_sd = None
            if len(runs) > 1:
                seed_sd = float(np.std(seed_means, ddof=1))
            summaries.append(dict(
                policy=name, mean=stats['mean'], ci95_low=interval[0], ci95_high=interval[1],
                n_hands=stats['n_hands'], eval_seeds=len(runs), eval_seed_sd=seed_sd,
                mean_eval_runtime_s=float(np.mean(runtimes)),
                mean_decision_runtime_s=float(np.mean(decision_times))))
            for c, cls in enumerate(pe.UCI_CLASS_NAMES):
                stat = summarize(per_hand[labels == c])
                interval = stat['ci95'] or [None, None]
                class_rows.append(dict(policy=name, starting_class=cls, n_hands=stat['n_hands'],
                                       mean=stat['mean'], ci95_low=interval[0], ci95_high=interval[1]))
    return summaries, class_rows, labels
