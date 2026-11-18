# Five-Card Draw Poker with Tabular Q-Learning (CSE 150A)

**Team:** Ryan Cao, Dustin Pham, Andrew Pham, Ganesh Kumarappan, Raymond Tran

## Status

| Task | Owner | Status |
|---|---|---|
| 1. Dataset and poker environment | Ganesh | ✅ Done (this commit) |
| 2. Q-learning agent | Dustin | — |
| 3. Baseline strategies | Andrew | — |
| 4. State abstraction and reward design | Raymond | — |
| 5. Evaluation and plots | Ryan | — |

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

## Useful tools for other tasks

- **Dustin (Q-learning):** a scalar episode takes about 18 µs. `pe.exact_action_values(hand)` gives the true Q*(s, ·) for all 32 actions by enumeration (a few seconds per hand), which is good for checking that the trainer converges to the right values.
- **Andrew (baselines):** `pe.parse_hand`, `pe.mask_from_keep`, `pe.mask_to_str`, and `pe.exact_action_values` let you check heuristic decisions against the optimal action. For example, keeping JJ is worth 1.537 and keeping two pair 2.596.
- **Raymond (abstraction/rewards):** `pe.CATEGORY_NAMES` and `pe.JACKS_OR_BETTER` define the paytable format, and `pe.payout_batch` takes any paytable.
- **Ryan (evaluation):** `pe.simulate_draw_batch` and `pe.payout_batch` run about 2 µs per hand, and `load_split("eval")` includes `uci_class` for per-class breakdowns.

## Rebuilding the split (optional)

The split is already committed. To rebuild it, download the UCI Poker Hand dataset (`poker+hand.zip`), extract it into `data/raw/`, and run `python poker_data.py`. The rebuild is deterministic with seed 150.
