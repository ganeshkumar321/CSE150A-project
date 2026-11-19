"""Alternative paytables and reward transforms (Task 4: reward design).

Every reward in this project is a function of the final hand's payout category,
so each one is a plain paytable dict over the 11 keys in
``poker_env.CATEGORY_NAMES`` and works with ``pe.payout`` / ``pe.payout_batch``.

Two kinds of table live here:

* **Games** (``GAMES``): different payout schedules a player could face. The
  agent's objective is the expected payout under that table.
* **Training rewards** (``TRAINING_REWARDS``): transforms of the main 9/6 table
  that keep the order of the categories but shrink the jackpot. They lower the
  variance of the learning signal, but they change which action has the
  highest expected reward, so a policy trained on them is always *scored* on
  the main table.

Use ``get_paytable(name)`` to look one up by name.
"""

from __future__ import annotations

import math

import poker_env as pe

# 9/6 Jacks or Better, the main table (full-pay; royal = 800 per unit bet, the
# max-bet rate). Same object as pe.JACKS_OR_BETTER.
JOB_9_6 = pe.JACKS_OR_BETTER

# 8/5 Jacks or Better: the common casino table (full house 8, flush 5).
JOB_8_5 = {**JOB_9_6, "full_house": 8, "flush": 5}

# 9/6 with the royal at 250, the rate when betting less than max coins.
JOB_9_6_ROYAL250 = {**JOB_9_6, "royal_flush": 250}

# Binary reward: 1 for any paying hand (jacks or better and up), else 0.
# Its optimal policy maximizes the probability of being paid at all.
WIN_ONLY = {name: (0 if name in ("nothing", "low_pair") else 1) for name in pe.CATEGORY_NAMES}


def transform(paytable: dict, fn) -> dict:
    """Apply ``fn`` to every payout of a paytable."""
    return {name: float(fn(v)) for name, v in paytable.items()}


# Monotone transforms of the main table, for training only.
JOB_9_6_LOG1P = transform(JOB_9_6, math.log1p)          # 800 -> 6.69, 1 -> 0.69
JOB_9_6_CLIP25 = transform(JOB_9_6, lambda v: min(v, 25))  # straight/royal flush -> 25

GAMES = {
    "job_9_6": JOB_9_6,
    "job_8_5": JOB_8_5,
    "job_9_6_royal250": JOB_9_6_ROYAL250,
    "win_only": WIN_ONLY,
}
TRAINING_REWARDS = {
    "job_9_6_log1p": JOB_9_6_LOG1P,
    "job_9_6_clip25": JOB_9_6_CLIP25,
}
PAYTABLES = {**GAMES, **TRAINING_REWARDS}

MAIN = "job_9_6"


def get_paytable(name: str) -> dict:
    try:
        return PAYTABLES[name]
    except KeyError:
        raise KeyError(f"unknown paytable {name!r}; choose from {sorted(PAYTABLES)}") from None


def check_paytable(paytable: dict) -> None:
    """Raise if a paytable is missing keys, has extra keys, or is not monotone."""
    keys = set(paytable)
    want = set(pe.CATEGORY_NAMES)
    if keys != want:
        raise ValueError(f"paytable keys differ: missing {want - keys}, extra {keys - want}")
    vals = [paytable[name] for name in pe.CATEGORY_NAMES]
    if any(b < a for a, b in zip(vals, vals[1:])):
        raise ValueError("paytable payouts must be non-decreasing in hand strength")
