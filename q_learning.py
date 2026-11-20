"""Tabular Q-learning for a single draw, with actions in canonical card order."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

import poker_env as pe

StateEncoder = Callable[[Sequence[int]], tuple[int, Sequence[int]]]


def canonical_state(hand) -> tuple[int, Sequence[int]]:
    """Return the state ID and mapping from canonical to original positions."""
    _, perm, state = pe.canonicalize(hand)
    return state, perm


class QLearningAgent:
    """Learn expected payouts for 32 keep masks; calling the agent is greedy.

    Custom encoders return (state_id, perm), where perm[j] is the original
    position of encoded card j. Give each encoding a distinct state_encoding name.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        alpha_decay: float = 1.0,
        alpha_min: float = 0.0,
        epsilon: float = 1.0,
        epsilon_decay: float = 1e-6,
        epsilon_min: float = 0.05,
        initial_q: float = 0.0,
        seed: int = 0,
        n_states: int = pe.N_CANONICAL_STATES,
        state_encoding: str = "canonical",
        encoder: StateEncoder | None = None,
    ):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.initial_q = initial_q
        self.seed = seed
        self.n_states = n_states
        self.state_encoding = state_encoding
        self.encoder = canonical_state if encoder is None else encoder

        shape = (n_states, pe.N_ACTIONS)
        self.Q = np.full(shape, initial_q, dtype=np.float64)
        self.N = np.zeros(shape, dtype=np.int64)

        self.episode = 0
        self.rng = np.random.default_rng(seed)
        self.metadata = {}
        self.history = []
        self.runtime_s = 0.0

    def get_config(self) -> dict:
        """Return constructor settings for checkpoints and experiment records."""
        return dict(n_states=self.n_states, alpha=self.alpha, alpha_decay=self.alpha_decay,
                    alpha_min=self.alpha_min, epsilon=self.epsilon,
                    epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min,
                    initial_q=self.initial_q, seed=self.seed, state_encoding=self.state_encoding)

    def exploration_rate(self) -> float:
        """Return epsilon for the next training episode."""
        return max(self.epsilon_min, self.epsilon / (1 + self.epsilon_decay * self.episode))

    def select_action(self, state: int, *, explore: bool = True) -> int:
        """Choose an encoded mask; randomize greedy ties only during training."""
        values = self.Q[state]
        if not explore:
            return int(np.argmax(values))

        # explore across all keep masks
        if self.rng.random() < self.exploration_rate():
            return int(self.rng.integers(pe.N_ACTIONS))

        # distribute training among tied actions
        ties = np.flatnonzero(values == values.max())
        return int(self.rng.choice(ties))

    def __call__(self, hand) -> int:
        """Return a greedy keep mask in original positions without consuming RNG."""
        state, perm = self.encoder(hand)
        action = self.select_action(state, explore=False)
        return pe.canonical_mask_to_original(action, perm)

    def update(self, state: int, action: int, reward: float) -> None:
        """Apply the terminal update; N counts visits and episode counts updates."""
        # estimate the terminal expected payout
        visits = int(self.N[state, action])
        alpha = max(self.alpha_min, self.alpha / (1 + self.alpha_decay * visits))
        self.Q[state, action] += alpha * (reward - self.Q[state, action])
        self.N[state, action] += 1
        self.episode += 1

    def save(self, path: str | Path) -> None:
        """Atomically save arrays, settings, RNG, and training records as NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(version=2, rng=self.rng.bit_generator.state, metadata=self.metadata,
                     history=self.history, runtime_s=self.runtime_s)

        # replace only after successful serialization
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, Q=self.Q, N=self.N, episode=self.episode,
                                    config=json.dumps(self.get_config()), state=json.dumps(state))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, encoder: StateEncoder | None = None) -> QLearningAgent:
        """Restore a checkpoint without pickle; custom encoders must be supplied."""
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"].item()))

            # require the saved state encoder
            if config["state_encoding"] != "canonical" and encoder is None:
                raise ValueError("loading custom states requires the original encoder")

            agent = cls(**config, encoder=encoder)
            state = json.loads(str(data["state"].item()))
            _validate_checkpoint(data, agent.n_states, state)
            agent.Q = data["Q"].copy()
            agent.N = data["N"].copy()
            agent.episode = int(data["episode"].item())

        # restore the next random draw
        agent.rng.bit_generator.state = state["rng"]
        agent.metadata = state["metadata"]
        agent.history = state["history"]
        agent.runtime_s = state["runtime_s"]
        return agent


def _validate_checkpoint(data, n_states: int, state: dict) -> None:
    """Check checkpoint format and table dimensions."""
    shape = (n_states, pe.N_ACTIONS)

    # check the checkpoint schema version
    if state["version"] != 2:
        raise ValueError("unsupported checkpoint version")

    # verify configured state action dimensions
    if data["Q"].shape != shape or data["N"].shape != shape:
        raise ValueError("checkpoint table dimensions disagree with config")


def train(agent: QLearningAgent, hands, episodes: int, *, simulator=pe.simulate_draw,
          reward_fn=pe.payout, paytable=None) -> dict:
    """Sample starting hands uniformly with replacement and update once per draw.

    Args:
        hands: Nonempty sequence of starting hands, in original position order.
        episodes: Additional updates, preserving the agent's global schedules.
        simulator: Callable (hand, original_mask, rng) -> final_hand.
        reward_fn: Callable (final_hand, paytable) -> scalar payout.

    Returns:
        Training-window statistics; these include exploration, not held-out evaluation.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    start = perf_counter()
    reward_sum = 0.0

    # learn from independent draw episodes
    for _ in range(episodes):
        hand = hands[int(agent.rng.integers(len(hands)))]
        state, perm = agent.encoder(hand)
        action = agent.select_action(state)
        mask = pe.canonical_mask_to_original(action, perm)

        reward = float(reward_fn(simulator(hand, mask, agent.rng), paytable))
        agent.update(state, action, reward)
        reward_sum += reward

    # record exploratory training window statistics
    agent.runtime_s += perf_counter() - start
    record = dict(episode=agent.episode, episodes=episodes, mean_reward=reward_sum / episodes,
                  epsilon=agent.exploration_rate(), runtime_s=agent.runtime_s)
    agent.history.append(record)
    return record
