"""Train Task 2 on the shared training split; never load held-out evaluation hands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import poker_env as pe
from q_learning import QLearningAgent, train

ROOT = Path(__file__).resolve().parent


def parse_args():
    """Parse new-run settings or resume an existing checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000, help="additional training episodes")
    parser.add_argument("--output", type=Path, required=True, help="run directory")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alpha-decay", type=float)
    parser.add_argument("--alpha-min", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--epsilon-decay", type=float)
    parser.add_argument("--epsilon-min", type=float)
    parser.add_argument("--initial-q", type=float)
    parser.add_argument("--paytable", type=Path, help="JSON dict with all 11 payout categories")

    # validate budgets before loading data
    args = parser.parse_args()
    if min(args.episodes, args.log_every, args.checkpoint_every) <= 0:
        parser.error("episode budgets and logging/checkpoint intervals must be positive")

    # preserve settings across resumed runs
    settings = (args.seed, args.alpha, args.alpha_decay, args.alpha_min, args.epsilon,
                args.epsilon_decay, args.epsilon_min, args.initial_q, args.paytable)
    if args.resume and any(value is not None for value in settings):
        parser.error("resume restores settings and paytable; omit configuration flags")
    return args


def new_agent(args) -> QLearningAgent:
    """Pass supplied CLI settings directly to the agent constructor."""
    names = ("seed", "alpha", "alpha_decay", "alpha_min", "epsilon", "epsilon_decay",
             "epsilon_min", "initial_q")
    settings = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    return QLearningAgent(**settings)


def load_training_data() -> tuple[np.ndarray, dict]:
    """Read only the shared train file and identify its exact bytes for resumption."""
    path = ROOT / "data" / "splits" / "train.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        hands = data["cards"].astype(np.int64)
    return hands, dict(split="train", split_sha256=digest, sampling="uniform_with_replacement")


def load_paytable(path: Path | None) -> dict:
    """Load payouts in the shared category order."""
    table = dict(pe.JACKS_OR_BETTER) if path is None else json.loads(path.read_text())
    return {key: float(table[key]) for key in pe.CATEGORY_NAMES}


def write_results(agent: QLearningAgent, output: Path) -> None:
    """Write training records with reserved fields for Task 5's held-out results."""
    result = dict(config=agent.get_config(), seed=agent.seed, episodes=agent.episode,
                  eval_mean=None, eval_ci95=None, by_class={}, runtime_s=agent.runtime_s,
                  metadata=agent.metadata, training_windows=agent.history)
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "results.json")


def main():
    """Train in reporting windows and save cumulative episode-numbered checkpoints."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hands, data_info = load_training_data()

    # restore or initialize training settings
    if args.resume:
        agent = QLearningAgent.load(args.resume)
        if any(agent.metadata.get(key) != value for key, value in data_info.items()):
            raise ValueError("resume requires the same shared training split and sampling rule")
        paytable = agent.metadata["paytable"]
    else:
        if (args.output / "results.json").exists() or any(args.output.glob("checkpoint_*.npz")):
            raise ValueError("run directory already contains results; use --resume or a new directory")
        agent = new_agent(args)
        paytable = load_paytable(args.paytable)
        agent.metadata = dict(data_info, paytable=paytable)

    target = agent.episode + args.episodes

    # train between scheduled checkpoint boundaries
    while agent.episode < target:
        until_checkpoint = args.checkpoint_every - agent.episode % args.checkpoint_every
        count = min(args.log_every, until_checkpoint, target - agent.episode)
        record = train(agent, hands, count, paytable=paytable)
        print(json.dumps(record), flush=True)

        # persist progress at checkpoint boundaries
        if agent.episode % args.checkpoint_every == 0 or agent.episode == target:
            agent.save(args.output / f"checkpoint_{agent.episode:09d}.npz")
        write_results(agent, args.output)


if __name__ == "__main__":
    main()
