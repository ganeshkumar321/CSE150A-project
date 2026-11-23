"""Task 5: held-out Monte Carlo evaluation of baseline and checkpoint policies.

python evaluate_policies.py --output results/task5
python evaluate_policies.py --checkpoints runs/seed0/checkpoint_000100000.npz --output results/task5-checkpoints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import baselines
import poker_env as pe
from poker_data import load_split, SPLIT_DIR
from q_learning import QLearningAgent
from evaluation_results import file_hash, summarize

# Task 3 owns the decision rules; evaluation only calls the policies.
BASELINES = dict(keep_all=baselines.keep_all, pairs=baselines.pairs_policy,
                 basic=baselines.basic_policy, strategy=baselines.strategy_policy)


def evaluate_policy(policy, hands, classes, *, seed, draws=10, batch_size=2000):
    """policy(hand)->original-position mask; do not train or consume training RNG.

    A stochastic policy is called afresh for each draw. Repeated outcomes of
    one starting hand are averaged before estimating uncertainty. Each seed
    starts an independent draw generator; policies share its initial seed.
    """
    hands = np.asarray(hands)
    classes = np.asarray(classes)
    if hands.ndim != 2 or hands.shape[1] != pe.HAND_SIZE or len(hands) == 0:
        raise ValueError("hands must be a nonempty (N,5) array")
    if (not np.issubdtype(hands.dtype, np.integer) or np.any((hands < 0) | (hands >= pe.N_CARDS))
            or np.any(np.diff(np.sort(hands, axis=1), axis=1) == 0)):
        raise ValueError('each hand must contain five distinct integer cards in 0..51')
    if classes.shape != (len(hands),) or not np.all(np.isin(classes, range(len(pe.UCI_CLASS_NAMES)))):
        raise ValueError("one UCI class in 0..9 is required per starting hand")
    if (not isinstance(draws, (int, np.integer)) or not isinstance(batch_size, (int, np.integer))
            or draws <= 0 or batch_size <= 0):
        raise ValueError("draws and batch_size must be positive")
    rng = np.random.default_rng(seed)
    totals = np.zeros(len(hands))
    decision_s = draw_score_s = 0.0
    start = perf_counter()
    for repeat in range(draws):
        for low in range(0, len(hands), batch_size):
            batch = hands[low:low + batch_size]
            tick = perf_counter()
            masks = []
            for hand in batch:
                mask = policy(hand.copy())
                if not isinstance(mask, (int, np.integer)) or not 0 <= mask < pe.N_ACTIONS:
                    raise ValueError("policy must return an integer keep mask in 0..31")
                masks.append(mask)
            decision_s += perf_counter() - tick
            tick = perf_counter()
            # Task 1 handles the original-position masks and the 47-card draw rule.
            final = pe.simulate_draw_batch(batch, masks, rng)
            payouts = pe.payout_batch(final, pe.JACKS_OR_BETTER)
            totals[low:low + len(batch)] += payouts
            draw_score_s += perf_counter() - tick
    per_hand = totals / draws
    overall = summarize(per_hand)
    by_class = {}
    for c, name in enumerate(pe.UCI_CLASS_NAMES):
        by_class[name] = summarize(per_hand[classes == c])
    return dict(eval_mean=overall['mean'], eval_ci95=overall['ci95'],
                se=overall['se'], n_hands=len(hands), draws_per_hand=draws,
                n_outcomes=len(hands) * draws, eval_seed=seed,
                by_class=by_class, runtime_s=perf_counter() - start,
                decision_runtime_s=decision_s, draw_score_runtime_s=draw_score_s,
                per_hand=per_hand)


def load_checkpoint(path):
    """Restore Task 2 canonical/ranks checkpoints with their original encoder."""
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved['config'].item()))
    encoder = None
    if config['state_encoding'] not in ('canonical', 'ranks'):
        raise ValueError('Task 2 checkpoints must use canonical or ranks states; '
                         'Task 4 coarse hold-rule actions have a different format')
    if config['state_encoding'] != 'canonical':
        from abstraction import get_abstraction
        encoder = get_abstraction(config['state_encoding']).state_encoder()
    agent = QLearningAgent.load(path, encoder=encoder)
    if not np.all(np.isfinite(agent.Q)) or np.any(agent.N < 0):
        raise ValueError('checkpoint must contain finite Q values and nonnegative visit counts')
    if encoder is None:
        expected_states = pe.N_CANONICAL_STATES
    else:
        expected_states = get_abstraction('ranks').n_states
    if agent.n_states != expected_states:
        raise ValueError('checkpoint row count does not match the saved state encoding')
    return agent


def evaluate_seeds(name, policy, hands, classes, args, records, arrays, metadata=None):
    """Use the same seed loop for a baseline or a loaded greedy agent."""
    for seed in args.seeds:
        current_policy = policy
        if name == 'random':
            current_policy = baselines.random_policy(np.random.SeedSequence([seed, 1]))
        result = evaluate_policy(current_policy, hands, classes, seed=seed,
                                 draws=args.draws, batch_size=args.batch_size)
        key = f'policy{len(records)}'
        arrays[key] = result.pop('per_hand')
        result.update(policy=name, values_key=key)
        if metadata:
            result.update(metadata)
        records.append(result)
        print(f"{name}, eval seed {seed}: {result['eval_mean']:.6f}", flush=True)


def save_results(output_dir, indices, records, arrays, args):
    """Save raw hand averages and the settings needed to reproduce the tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / 'per_hand.npz', indices=indices, **arrays)
    output = dict(schema_version=1, method='monte_carlo', scoring_paytable=dict(pe.JACKS_OR_BETTER),
                  split='eval', split_sha256=file_hash(Path(SPLIT_DIR) / 'eval.npz'),
                  subset_seed=args.subset_seed, batch_size=args.batch_size,
                  random_action_stream='SeedSequence([eval_seed, 1]); independent of draw RNG',
                  uncertainty='Normal 95% CI over starting-hand averages; repeated draws are clustered by hand.',
                  records=records)
    (output_dir / 'evaluation.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/task5'))
    parser.add_argument('--checkpoints', type=Path, nargs='*', default=[])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--draws', type=int, default=10)
    parser.add_argument('--hands', type=int, default=None, help='optional smoke-test subset size')
    parser.add_argument('--subset-seed', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=2000)
    args = parser.parse_args()
    if (args.draws <= 0 or args.batch_size <= 0 or len(set(args.seeds)) != len(args.seeds)
            or min(args.seeds) < 0 or args.subset_seed < 0):
        parser.error('positive draw/batch counts and unique seeds are required')
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            parser.error(f'checkpoint not found: {checkpoint}')
    if len({p.resolve() for p in args.checkpoints}) != len(args.checkpoints):
        parser.error('provide each checkpoint only once')
    return args


def main():
    args = parse_args()
    split = load_split('eval')
    indices = np.arange(len(split['cards']))
    if args.hands is not None:
        if not 1 <= args.hands <= len(indices):
            raise ValueError('--hands must be between 1 and the eval split size')
        indices = np.sort(np.random.default_rng(args.subset_seed).choice(indices, args.hands, replace=False))
    hands, classes = split['cards'][indices], split['uci_class'][indices]
    # Build shared scoring tables before timing policies (first-run cache creation is expensive).
    pe.payout_batch(hands[:1])
    records, arrays = [], {}
    evaluate_seeds('random', None, hands, classes, args, records, arrays)
    for name, policy in BASELINES.items():
        evaluate_seeds(name, policy, hands, classes, args, records, arrays)
    for checkpoint in args.checkpoints:
        name = checkpoint.with_suffix('').as_posix()
        agent = load_checkpoint(checkpoint)
        info = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                    config=agent.get_config(), seed=agent.seed, episodes=agent.episode,
                    training_runtime_s=agent.runtime_s, metadata=agent.metadata)
        evaluate_seeds(name, agent, hands, classes, args, records, arrays, info)
    save_results(args.output, indices, records, arrays, args)


if __name__ == '__main__':
    main()
