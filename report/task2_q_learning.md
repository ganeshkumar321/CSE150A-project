### Q-learning

Each episode contains one starting hand, one keep/discard decision, one replacement draw, and one terminal payout. We estimate the expected payout Q(s,a) for each state/action pair with

`Q(s,a) <- Q(s,a) + alpha(s,a) * (reward - Q(s,a))`.

There is no next-state maximum or discount factor because the draw ends the episode. Payouts are in units of the original bet, following Task 1's paytable; we do not subtract the bet. Subtracting a fixed bet would shift every action value equally without changing the best action.

**State and action ordering.** The default table has 134,459 canonical states and 32 actions. Both Q and visit counts N use `[state_id, canonical_mask]` indexing. The trainer canonicalizes the starting hand, selects a canonical mask, and maps that mask back to the original five positions before calling the simulator. The evaluator-facing agent is callable as `agent(hand) -> original_mask`. Alternative state encoders can return `(state_id, perm)` and select a different number of states through the agent’s `n_states` constructor argument; they must retain the same 32-action ordering within their chosen card order. The encoder is supplied separately when loading a custom-state checkpoint, and its name is stored in the configuration. A coarse encoder must use a consistent card order within each abstract state, so that action bits retain comparable meaning across hands.

**Learning rates.** A schedule has the form `max(minimum, initial / (1 + decay * step))`. For learning rates, `step` is the number of previous visits to that specific `(s,a)` pair. The visit count increases after the update. The default `initial=1, decay=1, minimum=0` therefore gives `alpha=1/N` on the Nth visit, making Q the sample mean of observed payouts for that pair. Constant rates use `decay=0`; these continue to weight recent observations and need not converge exactly under stochastic feedback. With sample-average updates the first observation replaces optimistic initialization, while a constant rate retains some of that initial value.

**Exploration.** Epsilon uses the number of completed episodes as its zero-based step. With probability epsilon the trainer chooses uniformly among all 32 actions; otherwise it chooses uniformly among the maximizers of Q. Random tie breaking prevents a zero-filled table from always choosing mask 0 during training. Greedy inference uses the lowest-numbered maximizing mask, performs no exploration, and does not advance the training RNG. Unvisited states consequently choose mask 0 at inference when all entries are equal; no heuristic fallback is built into the learned policy. Optimistic initialization is supported by filling Q with a configurable initial payout estimate. The proposed value 5 is optimistic relative to many expected payouts, but is not an upper bound: the largest terminal payout is 800.

### Training configurations

Starting hands are sampled uniformly **with replacement** from the shared `train.npz`, maintaining its empirical frequency over canonical states. We do not sample states uniformly, train on development hands, or open the evaluation split. Draws use Task 1's simulator and the 47-card exclusion rule. One `np.random.default_rng(seed)` controls starting-hand sampling, epsilon decisions, training ties, and replacement draws. Reproducibility assumes the same code, NumPy environment, input split, encoder, and paytable.

The proposed comparison uses seeds **0, 1, 2, 3, 4** and **5,000,000 episodes per configuration per seed**, with reporting windows of 10,000 episodes and checkpoints every 100,000 episodes. These are initial experiment budgets, not claims that every state/action pair is well estimated. There are 4,302,688 entries in the full table, states are unevenly represented, and rare payouts have high variance. A 100,000-episode run is useful for integration only. Task 5 should inspect coverage and variation across seeds before drawing performance conclusions. Tune on development hands before the final held-out comparison; the biased development class distribution should not be reported as the overall expected payout.

| Configuration | Alpha (initial, decay, minimum) | Epsilon (initial, decay, minimum) | Initial Q |
|---|---|---|---:|
| Default / sample average | (1, 1, 0) | (1, 0.000001, 0.05) | 0 |
| Constant learning rate | (0.1, 0, 0) | (1, 0.000001, 0.05) | 0 |
| Constant exploration | (1, 1, 0) | (0.2, 0, 0) | 0 |
| Optimistic initialization | (1, 1, 0) | (1, 0.000001, 0.05) | 5 |

The default epsilon reaches 1/6 after five million episodes; its 0.05 floor is reached at nineteen million. These choices are explicit starting configurations, not tuned results. For Task 4 comparisons, replace the paytable or encoder while holding the other settings, episode budget, and seed list fixed. Equal seeds provide reproducible runs, but do not guarantee identical draw streams across policies: different actions can consume different numbers of random variates.

### Checkpoints and results

Each compressed NPZ contains `Q` (float64), `N` (int64), scalar `episode`, JSON-string `config`, and JSON-string `state`. The first four fields match Task 1's proposal. `config` is a plain dictionary of constructor arguments, including the scalar rate/decay/minimum settings, initialization, seed, state count, and encoder name. `state` version 2 additionally stores the generator state, training history, accumulated training-loop runtime in seconds, and metadata. CLI metadata includes the complete paytable, sampling rule, and SHA-256 of the training split. A full table occupies about 65.7 MiB across Q and N before compression.

Checkpoints use a temporary file and atomic replacement. Loading requires no pickle. Resume restores the schedules at the global episode/local visit counts and the exact RNG state; the CLI rejects a changed training split or configuration overrides. The `--episodes` flag always means **additional** updates. Intermediate files are named `checkpoint_000100000.npz`, etc.; the final episode also receives a checkpoint even if it is between scheduled boundaries. The library caller is responsible for using the same simulator and reward implementation after resuming.

`results.json` contains `config`, `seed`, cumulative `episodes`, `runtime_s`, `metadata`, and `training_windows`. Each training window records its ending episode, number of episodes, mean exploratory payout, epsilon at the next episode, and cumulative runtime. Runtime excludes loading, checkpoint compression, and result-file writing. For Task 5 compatibility it also reserves `eval_mean: null`, `eval_ci95: null`, and `by_class: {}`. These fields deliberately contain no fabricated measurements. Training-window payout is not a greedy held-out learning curve. Task 5 can load any saved checkpoint as a callable policy and evaluate it on the shared held-out hands.

### Verification

`python -m unittest test_q_learning -v` checks the terminal update, visit counts, per-pair sample-average rates, constant/decaying schedules, optimistic initialization, epsilon exploration, training tie breaking, and deterministic inference. A two-state synthetic environment uses the same simulator/reward interface with 32 actions: masks 7 and 19 respectively have Bernoulli mean 0.9, and all other actions have mean 0.2. After 64,000 seeded episodes with uniform exploration, all 64 estimates are within 0.06 of their known means and greedy policies recover both optimal actions. This verifies the trainer independently of the poker reward distribution.

Checkpoint tests compare 3,000 uninterrupted synthetic episodes against 1,100 episodes followed by save/load and 1,900 more. Q, N, episode, configuration, and generator state match exactly. Poker integration tests verify pair-keeping across all 120 card-position permutations, inspect the original-position mask passed to the simulator, and learn the known payout of 800 for keeping a royal flush. The existing Task 1 test suite is also retained and run. Poker CLI smoke checks verify train-split loading, full-sized checkpoints, JSON output, and deterministic resumption; they are integration checks rather than final performance experiments.
