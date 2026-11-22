"""Behavioral tests for Task 2; run with python -m unittest test_q_learning -v."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import poker_env as pe
from q_learning import QLearningAgent, train


def synthetic_encoder(hand):
    """Use the first input as a state ID, with identity action positions."""
    return int(hand[0]), tuple(range(5))


def synthetic_draw(hand, mask, rng):
    """Return Bernoulli feedback: one arm per state pays 0.9, the others 0.2."""
    best = (7, 19)[int(hand[0])]
    return float(rng.random() < (0.9 if mask == best else 0.2))


def synthetic_reward(final_hand, paytable):
    """Use synthetic feedback directly while preserving the reward interface."""
    return final_hand


def synthetic_agent(**settings):
    """Build a tiny agent so learning tests do not depend on poker tables."""
    settings.setdefault("epsilon_min", 0.0)
    settings.setdefault("epsilon_decay", 0.0)
    return QLearningAgent(n_states=2, state_encoding="synthetic",
                          encoder=synthetic_encoder, **settings)


def train_synthetic(agent, episodes):
    """Exercise the production training loop with the agreed one-step interfaces."""
    return train(agent, [[0], [1]], episodes, simulator=synthetic_draw,
                 reward_fn=synthetic_reward)


class QLearningTests(unittest.TestCase):
    def test_terminal_update_and_visit_counts(self):
        agent = synthetic_agent(alpha=0.25, alpha_decay=0, initial_q=4.0)
        agent.update(0, 3, 8.0)
        self.assertEqual(agent.Q[0, 3], 5.0)
        self.assertEqual(agent.Q[1, 3], 4.0)
        self.assertEqual(agent.N.sum(), 1)
        self.assertEqual(agent.episode, 1)

        agent.update(0, 3, 1.0)
        self.assertEqual(agent.Q[0, 3], 4.0)

    def test_sample_average_uses_local_visits(self):
        agent = synthetic_agent()

        # interleave visits across separate pairs
        for state, action, reward in [(0, 1, 2), (1, 5, 100), (0, 1, 4), (0, 1, 9)]:
            agent.update(state, action, reward)

        self.assertEqual(agent.Q[0, 1], 5.0)
        self.assertEqual(agent.Q[1, 5], 100.0)
        self.assertEqual(agent.N[0, 1], 3)

    def test_exploration_decay_and_floor(self):
        agent = synthetic_agent(epsilon=1, epsilon_decay=1, epsilon_min=0.1)
        agent.episode = 3
        self.assertEqual(agent.exploration_rate(), 0.25)
        agent.episode = 1000
        self.assertEqual(agent.exploration_rate(), 0.1)

    def test_exploration_and_greedy_policy(self):
        agent = synthetic_agent(epsilon=1, seed=4)
        agent.Q[0, 7] = 10
        self.assertEqual(agent([0]), 7)

        # check coverage under full exploration
        counts = np.bincount([agent.select_action(0) for _ in range(3200)], minlength=32)
        self.assertTrue((counts > 60).all())

        agent.epsilon = 0
        self.assertTrue(all(agent.select_action(0) == 7 for _ in range(100)))

    def test_training_ties_are_random_evaluation_is_reproducible(self):
        agent = synthetic_agent(epsilon=0, epsilon_min=0)
        actions = {agent.select_action(0) for _ in range(200)}
        self.assertGreater(len(actions), 25)

        # preserve randomness during greedy inference
        before = json.dumps(agent.rng.bit_generator.state)
        self.assertEqual(agent([0]), 0)
        self.assertEqual(json.dumps(agent.rng.bit_generator.state), before)

    def test_known_stochastic_rewards(self):
        agent = synthetic_agent(epsilon=1, seed=150)
        record = train_synthetic(agent, 64_000)

        # compare estimates against known means
        expected = np.full((2, 32), 0.2)
        expected[0, 7] = expected[1, 19] = 0.9
        np.testing.assert_allclose(agent.Q, expected, atol=0.06, rtol=0)
        self.assertEqual([agent([0]), agent([1])], [7, 19])
        self.assertEqual(agent.N.sum(), 64_000)
        self.assertEqual(record["episode"], 64_000)

    def test_optimistic_initialization_encourages_unvisited_actions(self):
        agent = synthetic_agent(initial_q=10, epsilon=0, epsilon_min=0)
        actions = []

        # exhaust optimism across unvisited actions
        for _ in range(32):
            action = agent.select_action(0)
            actions.append(action)
            agent.update(0, action, 0)

        self.assertEqual(set(actions), set(range(32)))
        self.assertTrue((agent.N[0] == 1).all())

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        settings = dict(seed=8, alpha=0.8, alpha_decay=0.1, alpha_min=0.05,
                        epsilon=0.9, epsilon_decay=0.001, epsilon_min=0.1, initial_q=2)
        uninterrupted = synthetic_agent(**settings)
        resumed = synthetic_agent(**settings)
        train_synthetic(uninterrupted, 3000)
        train_synthetic(resumed, 1100)
        resumed.metadata = {"paytable": {"example": 4}, "split": "synthetic"}

        # reload before completing remaining episodes
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            resumed.save(path)
            resumed = QLearningAgent.load(path, encoder=synthetic_encoder)

        self.assertEqual(resumed.metadata["split"], "synthetic")
        self.assertEqual(resumed.history[0]["episode"], 1100)
        train_synthetic(resumed, 1900)

        # compare learned state and randomness
        np.testing.assert_array_equal(resumed.Q, uninterrupted.Q)
        np.testing.assert_array_equal(resumed.N, uninterrupted.N)
        self.assertEqual(resumed.get_config(), uninterrupted.get_config())
        self.assertEqual(resumed.episode, uninterrupted.episode)
        self.assertEqual(resumed.rng.bit_generator.state, uninterrupted.rng.bit_generator.state)

    def test_custom_checkpoint_requires_encoder(self):
        agent = synthetic_agent()

        # prevent silently changing state representations
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            agent.save(path)
            with self.assertRaisesRegex(ValueError, "original encoder"):
                QLearningAgent.load(path)

    def test_training_requires_positive_budget(self):
        agent = synthetic_agent()

        # reject invalid training episode budgets
        for episodes in (0, -1):
            with self.assertRaises(ValueError):
                train_synthetic(agent, episodes)

        self.assertEqual(agent.episode, 0)
        self.assertEqual(agent.N.sum(), 0)


class PokerIntegrationTests(unittest.TestCase):
    def test_policy_keeps_the_pair_under_all_position_permutations(self):
        hand = pe.parse_hand("Jc 2h Jd 5s 8c")
        _, perm, state = pe.canonicalize(hand)
        agent = QLearningAgent(n_states=state + 1, epsilon=0, epsilon_min=0)
        canonical_mask = pe.original_mask_to_canonical(0b00101, perm)
        agent.Q[state, canonical_mask] = 5

        # check every original card ordering
        for order in itertools.permutations(range(5)):
            reordered = hand[list(order)]
            mask = agent(reordered)
            kept = {int(reordered[i]) for i in pe.kept_positions(mask)}
            self.assertEqual(kept, {pe.str_to_card("Jc"), pe.str_to_card("Jd")})

    def test_trainer_maps_action_before_simulating(self):
        hand = pe.parse_hand("Jc 2h Jd 5s 8c")
        _, perm, state = pe.canonicalize(hand)
        agent = QLearningAgent(n_states=state + 1, epsilon=0, epsilon_min=0)
        action = pe.original_mask_to_canonical(0b00101, perm)
        agent.Q[state, action] = 5
        observed = []

        def draw(starting_hand, mask, rng):
            observed.append(mask)
            return starting_hand

        train(agent, [hand], 1, simulator=draw)
        self.assertEqual(observed, [0b00101])
        self.assertEqual(agent.Q[state, action], 1)
        self.assertEqual(agent.N[state, action], 1)

    def test_keep_royal_flush_has_known_poker_reward(self):
        hand = pe.parse_hand("Th Jh Qh Kh Ah")
        state = pe.state_id(hand)
        agent = QLearningAgent(n_states=state + 1, epsilon=0, epsilon_min=0)
        agent.Q[state, pe.KEEP_ALL] = 1
        train(agent, [hand], 5)
        self.assertEqual(agent.Q[state, pe.KEEP_ALL], 800)
        self.assertEqual(agent.N[state, pe.KEEP_ALL], 5)

    def test_alternative_paytable_reaches_the_trainer(self):
        hand = pe.parse_hand("Th Jh Qh Kh Ah")
        state = pe.state_id(hand)
        agent = QLearningAgent(n_states=state + 1, epsilon=0, epsilon_min=0)
        agent.Q[state, pe.KEEP_ALL] = 1
        paytable = dict(pe.JACKS_OR_BETTER, royal_flush=80)

        # learn the alternative terminal payout
        record = train(agent, [hand], 5, paytable=paytable)
        self.assertEqual(agent.Q[state, pe.KEEP_ALL], 80)
        self.assertEqual(record["mean_reward"], 80)


if __name__ == "__main__":
    unittest.main()
