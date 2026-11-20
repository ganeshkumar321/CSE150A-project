# Shared conventions (proposed by Task 1 — please confirm or object)

Everything below is implemented in `poker_env.py` / `poker_data.py`.

## Cards
- `card = 4*rank + suit`, an int in 0..51.
  - rank 0..12 = `2 3 4 5 6 7 8 9 T J Q K A`. The ace plays high, and also low in A-2-3-4-5.
  - suit 0..3 = `h s d c` (same order as UCI suits 1..4).
- Helpers: `pe.parse_hand("Ah Kd 7c 7s 2h")`, `pe.hand_to_str(hand)`, `pe.uci_to_cards(S, C)`.
- A **hand** is a length-5 array/sequence of distinct card ints. **Position i = UCI card i+1.** Hands keep their dealt order, and the simulator never reorders positions.

## Actions
- An int **mask in 0..31**. Bit `i` (`1 << i`) refers to hand position `i`, and **1 = keep**.
- `31` = keep all (stand pat), `0` = discard all.
- Helpers: `pe.mask_from_keep([0, 2])`, `pe.kept_positions(mask)`, `pe.mask_to_str(mask)` (`'KDKDD'`, position 0 first).

## Canonicalization
- `canon_hand, perm, state_id = pe.canonicalize(hand)`
  - `canon_hand` is a sorted tuple of 5 cards after suit relabeling (the lexicographically smallest over the 24 suit permutations).
  - `perm[j]` = **original position** of canonical card `j`.
  - `state_id` is in `0..134458` and indexes `pe.canonical_states()`.
- A mask chosen on the canonical hand → `pe.canonical_mask_to_original(mask, perm)` (bit `j` → bit `perm[j]`).
- The reverse direction is `pe.original_mask_to_canonical(mask, perm)`.
- `pe.canonical_policy(f)` wraps `f(canon_hand, state_id) -> canonical mask` into `policy(hand) -> mask`.
- The Q-table has shape `(134459, 32)` and is indexed by `[state_id, canonical_mask]`.

## Interfaces
```python
policy(hand) -> mask                                  # mask in original positions
pe.simulate_draw(hand, mask, rng) -> final_hand       # rng = np.random.Generator
pe.payout(final_hand, paytable=pe.JACKS_OR_BETTER) -> float
# vectorized (for evaluation)
pe.simulate_draw_batch(hands, masks, rng); pe.payout_batch(final_hands, paytable)
# ground truth for checking (slow: seconds per hand)
pe.exact_action_value(hand, mask, paytable); pe.exact_action_values(hand, paytable)
```
Replacement cards come uniformly, without replacement, from the 47 cards outside the **original** hand. Discards never return to the deck.

## Rewards
- A paytable is a dict over the 11 keys in `pe.CATEGORY_NAMES`:
  `nothing, low_pair, jacks_or_better, two_pair, three_of_a_kind, straight, flush, full_house, four_of_a_kind, straight_flush, royal_flush`.
- Main table is `pe.JACKS_OR_BETTER` (9/6): 0, 0, 1, 2, 3, 4, 6, 9, 25, 50, 800.
- Task 4 owns the alternative paytables, which use the same dict format.

## Data
- `from poker_data import load_split; s = load_split("train")` returns `cards (N,5)`, `uci_class`, `state_id`, `source_row`.
- **train** 771,352 / **dev** 24,898 / **eval** 50,000 hands. Splits are disjoint as unordered card sets and have no duplicates. See `data/splits/manifest.json`.
- train and eval come from `poker-hand-testing.data` (unbiased). dev is the deduplicated `poker-hand-training-true.data` (royal/straight flushes oversampled), used **only** for development and tuning.
- Rebuild with `python poker_data.py` (seed 150; deterministic).

## Experiments (Task 2 implemented; Task 5 evaluation pending)
- Seeds: split seed `150`. Run seeds `0, 1, 2, 3, 4`, each passed as `np.random.default_rng(seed)`.
- Checkpoints: `.npz` with `Q (134459,32) float64`, `N (134459,32) int64`, scalar `episode`, `config` (JSON string), and `state` (JSON string, schema version 2). Custom state encoders may change the row count, retaining 32 canonical-position actions.
- `config` is a plain dict of agent constructor arguments: `n_states`, `state_encoding`, `seed`, `initial_q`, `alpha`, `alpha_decay`, `alpha_min`, `epsilon`, `epsilon_decay`, and `epsilon_min`. Rates evaluate to `max(minimum, initial/(1+decay*step))`. Alpha's step is previous pair visits; epsilon's step is completed episodes.
- `state` preserves RNG state, metadata, training-window history, and training-loop runtime. CLI metadata records the training split SHA-256, sampling rule, and complete paytable. Resume restores all settings and consumes the same RNG stream.
- Default training: sample train hands uniformly with replacement; sample-average alpha `(1,1,0)`, epsilon `(1,1e-6,0.05)`, Q initialization 0. Proposed budget: 5M episodes per seed/configuration, reports every 10k, checkpoints every 100k plus the final episode. See `report/task2_q_learning.md` for the comparison settings and limits.
- Results: one JSON per run with `config`, `seed`, `episodes`, `eval_mean`, `eval_ci95`, `by_class`, `runtime_s`, `metadata`, `training_windows`. Task 2 leaves evaluation fields null/empty for Task 5. Training means include exploration and are not held-out learning curves.
- Greedy policies use the lowest-numbered maximizing mask and do not advance the training RNG. During training, maximizing ties are chosen uniformly at random.
