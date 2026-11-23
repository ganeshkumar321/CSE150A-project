# Five-Card Draw Poker with Tabular Q-Learning (CSE 150A)

## Task 5 evaluation

Task 5 is separate from the environment, learner, and baseline implementations:

| File | Role |
|---|---|
| `evaluate_policies.py` | Runs the existing policies/simulator and loads Task 2 checkpoints. |
| `evaluate_checkpoints.py` | Scores Task 2 checkpoints using Task 4's exact evaluator. |
| `evaluation_results.py` | Calculates shared statistics and builds comparison tables. |
| `plot_evaluation.py` | Draws figures from saved results and tables. |
| `test_evaluation.py` | Checks the evaluation pipeline and compatibility with the other tasks. |

Run from this repository with NumPy, pandas and matplotlib installed:

```powershell
python -m unittest test_evaluation -v
python evaluate_policies.py --output results/task5
python plot_evaluation.py
```

The final Task 2 exact-score records are in `results/task5-greedy/`. They contain
four configurations, five training seeds, and checkpoints at 1–5 million episodes.
`plot_evaluation.py` also regenerates their table and learning curve. These exact
scores are labeled separately from the Monte Carlo baseline results. To train
the 20 runs from the committed split and reproduce their exact scores, run:

```powershell
python run_task2_experiments.py --output runs/task2-reproduction
python evaluate_checkpoints.py --runs runs/task2-reproduction --output runs/task5-reproduction
python plot_evaluation.py --greedy-results runs/task5-reproduction/checkpoints.json --output runs/task5-reproduction/figures
```

Use a fresh score output directory for a new sweep. The training script resumes
from the latest checkpoint in each run directory. To regenerate the committed
figures from the saved scores, run `python plot_evaluation.py`.

Checkpoints and local training logs are not committed. Training settings and
commands are documented in `report/task2_q_learning.md` and the Task 2 section below.

The default evaluates the five Task 3 baselines on all 50,000 eval hands,
10 draws per hand, using evaluation seeds 0–4. For a quick check, use
`--hands 100 --draws 2 --seeds 0 --output results/task5-smoke`.
Point `plot_evaluation.py --evaluation results/task5-smoke/evaluation.json
--output report/figures/task5-smoke` at that smoke output; do not report it as
the full experiment.

Evaluate Task 2 saved checkpoints with the same runner:

```powershell
python evaluate_policies.py --checkpoints runs/seed0/checkpoint_000100000.npz runs/seed0/checkpoint_000200000.npz --output results/task5-checkpoints
python plot_evaluation.py --evaluation results/task5-checkpoints/evaluation.json --output report/figures/task5-checkpoints
```

Supply the actual paths from the team's training runs. Canonical and ranks
encoders are restored automatically. The runner uses greedy actions and never
updates the agent or consumes its training RNG. Multiple checkpoint episodes
produce a checkpoint learning-curve figure. Evaluate multiple training seeds
to assess training variability; evaluation seeds alone measure simulation noise.

Outputs are JSON records, per-hand averages (`.npz`), CSV comparison tables,
and PNG figures. The existing Task 4 JSON files supply a separately labeled
exact-score learning curve (main training reward only), not newly trained
policies. Task 4's batched learner is distinct from Task 2's agent.
See [Task 5 methodology and results](report/task5_evaluation.md).

**Team:** Ryan Cao, Dustin Pham, Andrew Pham, Ganesh Kumarappan, Raymond Tran

## Status

| Task | Owner | Status |
|---|---|---|
| 1. Dataset and poker environment | Ganesh | ✅ Done (this commit) |
| 2. Q-learning agent | Dustin | ✅ Implemented and tested; configuration results in Task 5 |
| 3. Baseline strategies | Andrew | ✅ Done |
| 4. State abstraction and reward design | Raymond | ✅ Done (see below) |
| 5. Evaluation and plots | Ryan | Baseline Monte Carlo evaluation and Task 2 exact checkpoint comparisons complete |

## What's in the repo (Task 1)

| File | What it is |
|---|---|
| `poker_env.py` | The poker environment: card encoding, hand scoring, payouts, canonicalization and mask mapping, the draw simulator, and exact expected values. **Import this from your task.** |
| `poker_data.py` | Loads the UCI data, deduplicates it, builds the train/dev/eval split, and provides `load_split(name)`. |
| `data/splits/` | **The shared split** (`train.npz`, `dev.npz`, `eval.npz`, `manifest.json`). It is already built, so just load it. |
| `CONVENTIONS.md` | Proposed shared conventions for cards, actions, canonicalization, interfaces, paytable, seeds, and checkpoint and result formats. **Please read it and say if you disagree.** |
| `poker.ipynb` | Validates every data claim in the proposal and tests the environment, with plots and tables. |
| `test_poker_env.py` | Unit tests: `python test_poker_env.py` |
| `report/task1_data_environment.md` | Report write-up for data processing and the environment. |
| `baselines.py` | Task 3: random, keep-all, pairs, basic and strategy heuristics, and an exact oracle, all as `policy(hand) -> mask`. `python baselines.py` prints the representative-hand table. |
| `baselines_check.py` | Exact comparison of every baseline against the optimum on sampled eval hands. |
| `test_baselines.py` | Unit tests: `python test_baselines.py` |
| `report/task3_baselines.md` | Report write-up for the baseline strategies. |

## What's in the repo (Task 4)

| File | What it is |
|---|---|
| `paytables.py` | Alternative paytables (8/5, royal = 250, win-only) and training-reward transforms (log1p, clip at 25), all in the shared dict format. `get_paytable(name)`. |
| `exact_values.py` | Exact Q*(s, a) for **all** 134,459 × 32 state-action pairs in about 4 s, for any paytable. Scores any policy with no Monte Carlo noise. |
| `abstraction.py` | Three state abstractions behind one interface: `canonical` (134,459 × 32), `ranks` (6,175 × 32), `coarse` (484 states × 11 hold rules). |
| `task4_experiments.py` | Exact analysis plus the abstraction and reward training sweeps. Writes `results/task4/` and `report/figures/task4_*.png`. |
| `test_task4.py` | Unit tests: `python test_task4.py` |
| `report/task4_abstraction_rewards.md` | Report write-up for state abstraction and reward design. |

- **Dustin (Q-learning):** `ev.q_table()[state_id]` is the true Q* row for any state, so you can measure how far your table is from it. `canonical` and `ranks` plug straight into `QLearningAgent(encoder=ab.state_encoder(), n_states=ab.n_states, state_encoding=ab.name)`. `coarse` uses 11 hold rules instead of 32 masks, so it goes through `z, sid, perm = ab.encode(hand)` and `mask = ab.to_mask(a, sid, perm)` with a `(ab.n_states, ab.n_actions)` table (as in `task4_experiments.py`).
- **Andrew (baselines):** `ev.optimal_masks()` is the exact optimal policy (0.995439 on a uniform deal), a ceiling for every baseline.
- **Ryan (evaluation):** `ev.expected_return(canon_masks, states=eval["state_id"])` gives a policy's exact eval score; `ab.greedy_canonical_masks(Q)` turns a Q-table into per-state masks.

## Quick start

Requires Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`.

```python
import numpy as np
import poker_env as pe
from poker_data import load_split

train = load_split("train")          # dict: cards (N,5), uci_class, state_id, source_row
hand = train["cards"][0]
print(pe.hand_to_str(hand))          # e.g. "Kh 7c Ks 2d Jh"

# Canonical state for the Q-table, and mapping a canonical mask back to the real hand
canon, perm, state_id = pe.canonicalize(hand)
canon_mask = 0b11000                         # keep canonical positions 3 and 4
mask = pe.canonical_mask_to_original(canon_mask, perm)

# One episode
rng = np.random.default_rng(0)
final = pe.simulate_draw(hand, mask, rng)
reward = pe.payout(final)                    # main paytable: pe.JACKS_OR_BETTER
```

The first call to anything that uses lookup tables builds `data/cache/tables_v1.npz`. This takes about 7 seconds and happens once per machine; the file is git-ignored.

## Key conventions (details in `CONVENTIONS.md`)

- **Card** = `4*rank + suit`, where rank 0..12 = 2..A and suit 0..3 = h, s, d, c. Use `pe.parse_hand("Ah Kd 7c 7s 2h")` to build hands by hand.
- **Action** = a mask in 0..31. Bit `i` refers to hand position `i`, and **1 = keep**. `31` keeps all cards; `0` discards all.
- **State** = `state_id` in 0..134,458 from `pe.canonicalize`. The Q-table is `(134459, 32)`, indexed by `[state_id, canonical_mask]`. Always map the chosen canonical mask back with `pe.canonical_mask_to_original(mask, perm)` before calling the simulator.
- **Interfaces:** `policy(hand) -> mask`, `pe.simulate_draw(hand, mask, rng) -> final_hand`, `pe.payout(final_hand, paytable) -> float`.
- **Paytable** = a dict over the 11 names in `pe.CATEGORY_NAMES`. Raymond: alternative paytables should use the same keys.

## Data split

| Split | Hands | Source | Use |
|---|---:|---|---|
| train | 771,352 | UCI test file (1M rows, unbiased) | starting hands for Q-learning |
| dev | 24,898 | UCI training file (25k rows, oversamples rare hands) | debugging and tuning only |
| eval | 50,000 | UCI test file, stratified by class, seed 150 | held-out evaluation |

The data is deduplicated on the unordered 5-card set, and the three splits share no hands.

## Findings worth knowing

- Every data claim in the proposal checks out except one. The training file oversamples **straight flushes by 14.4×, not "≈16×"** (the UCI docs say 14.43×). Please use 14.4× in the report.
- Our hand evaluator reproduces all 1,025,010 UCI labels, and the canonical state count is 134,459, both confirmed by enumeration.
- **The eval split has no straight flushes or royal flushes.** By chance we would expect fewer than 1 in 50k hands. Ryan: use constructed hands (e.g. `pe.parse_hand("Th Jh Qh Kh Ah")`) for per-class results on those two classes.
- 99.35% of eval hands share a canonical state with some train hand. This is intended: a tabular agent can only act well on states it has seen.

## Baselines (Task 3)

```python
import baselines as bl
bl.POLICIES            # name -> factory; call it to get policy(hand) -> mask
pol = bl.strategy_policy                       # best heuristic (16 priority rules)
mask, rule = bl.explain(hand, bl.STRATEGY_RULES)   # which rule fired
rand = bl.random_policy(seed=0)                # seeded; independent of the draw rng
oracle = bl.oracle_policy(paytable)            # exact optimum, ~0.1 s per new state
```

Exact mean payout on 3,000 eval hands: random 0.340, keep all 0.341, pairs 0.880, basic 0.929, **strategy 1.002**, oracle 1.004. The strategy policy makes the optimal hold on 95.3% of hands. Details are in `report/task3_baselines.md`.

- **Ryan:** the policies are plain functions, so they plug straight into the runner. The oracle is slow the first time it sees each state, so run it on a subsample or warm its cache once.
- **Raymond:** the heuristics are tuned for `pe.JACKS_OR_BETTER`. `oracle_policy(paytable)` stays optimal under alternative paytables.
- **Dustin:** `bl.regret(hand, mask)` gives the exact EV lost by any action, which helps when checking a trained Q-table's greedy policy.

## Useful tools for other tasks

- **Dustin (Q-learning):** a scalar episode takes about 18 µs. `pe.exact_action_values(hand)` gives the true Q*(s, ·) for all 32 actions by enumeration (a few seconds per hand), which is good for checking that the trainer converges to the right values.
- **Andrew (baselines):** `pe.parse_hand`, `pe.mask_from_keep`, `pe.mask_to_str`, and `pe.exact_action_values` let you check heuristic decisions against the optimal action. For example, keeping JJ is worth 1.537 and keeping two pair 2.596.
- **Raymond (abstraction/rewards):** `pe.CATEGORY_NAMES` and `pe.JACKS_OR_BETTER` define the paytable format, and `pe.payout_batch` takes any paytable.
- **Ryan (evaluation):** `pe.simulate_draw_batch` and `pe.payout_batch` run about 2 µs per hand, and `load_split("eval")` includes `uci_class` for per-class breakdowns.

## Rebuilding the split (optional)

The split is already committed. To rebuild it, download the UCI Poker Hand dataset (`poker+hand.zip`), extract it into `data/raw/`, and run `python poker_data.py`. The rebuild is deterministic with seed 150.

## Task 2: Q-learning

`q_learning.py` provides `QLearningAgent` and `train`. Training settings are
ordinary constructor arguments, such as `QLearningAgent(alpha=0.1, alpha_decay=0)`.
`train_q_learning.py` trains on the shared training split and writes checkpoints
and training statistics. The trainer itself requires only NumPy. Report text,
schedule definitions, and proposed experiment budgets are in
[`report/task2_q_learning.md`](report/task2_q_learning.md).

Run from this directory:

```sh
python -m unittest test_q_learning -v
python test_poker_env.py

# 100k-episode integration run; use a fresh output directory
python train_q_learning.py --episodes 100000 --seed 0 --output /tmp/poker-q-smoke

# five-million-episode default configuration, one of seeds 0..4
python train_q_learning.py --episodes 5000000 --seed 0 --output /tmp/poker-q-default-seed0

# resume for one million additional episodes, restoring all settings and RNG state
python train_q_learning.py --resume /tmp/poker-q-default-seed0/checkpoint_005000000.npz \
    --episodes 1000000 --output /tmp/poker-q-default-seed0
```

The proposed comparisons change one setting from the default per run:

| Variant | Additional CLI flags |
|---|---|
| Constant alpha 0.1 | `--alpha 0.1 --alpha-decay 0` |
| Constant epsilon 0.2 | `--epsilon 0.2 --epsilon-decay 0 --epsilon-min 0` |
| Optimistic initialization | `--initial-q 5` |

Use `--paytable path.json` for a complete alternative table with the 11 shared
category keys. Logging defaults to every 10,000 episodes; checkpoints default to
every 100,000 episodes and the final episode. Q is indexed by canonical masks;
the callable policy returns masks in original positions:

```python
from q_learning import QLearningAgent

policy = QLearningAgent.load("/tmp/poker-q-smoke/checkpoint_000100000.npz")
mask = policy(hand)  # greedy; no exploration or training RNG changes
```

For custom state representations, supply an encoder returning `(state_id, perm)`
and constructor arguments `n_states` and a distinct `state_encoding` name. Pass
the same encoder to `QLearningAgent.load(path, encoder=...)`. The library `train`
accepts simulator/reward callables and a paytable for synthetic tests or Task 4.

`results.json` stores exploratory training-window payouts. Held-out evaluation
fields are reserved as null/empty in the training logs. The held-out scores
are reported in [Task 5](report/task5_evaluation.md).
