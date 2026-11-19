"""Exact action values Q*(s, a) for all 134,459 canonical states at once.

``pe.exact_action_values`` enumerates every draw for one hand and takes seconds
per hand. This module gets the whole (134459, 32) table for a paytable in a few
seconds, which makes it possible to score any policy exactly (no Monte Carlo
noise), compute the best policy an abstraction can express, and compare
paytables.

Method
------
For a set X of k cards let S_k[X] = sum of payouts over all 5-card hands that
contain X. These sums come from one pass over the 2,598,960 hands (32 bincounts,
one per subset of positions). For a hand that keeps H and discards D (|D| = d),
the final hand is H plus d cards from the 47 outside the original hand. By
inclusion-exclusion over the discarded cards,

    sum over reachable final hands = sum over T subset of D of (-1)^|T| S[H u T]

and there are C(47, d) reachable final hands, so Q*(s, a) is that sum divided by
C(47, d). Over the 32 masks this is a signed superset sum, done as one matrix
product.

The values match ``pe.exact_action_values`` to machine precision (see
test_task4.py), and the optimal 9/6 return comes out to 0.995439, the published
value for full-pay Jacks or Better.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np

import poker_env as pe

_POPCOUNT = np.array([bin(m).count("1") for m in range(32)])
# Signed superset matrix: _MOBIUS[m, u] = (-1)^(|u|-|m|) if u contains m, else 0.
_MOBIUS = np.array([[(-1) ** (_POPCOUNT[u] - _POPCOUNT[m]) if (u & m) == m else 0
                     for u in range(32)] for m in range(32)], dtype=np.float64)
# Number of reachable final hands for each mask: C(47, number of discards).
_N_DRAWS = np.array([comb(47, 5 - _POPCOUNT[m]) for m in range(32)], dtype=np.float64)


def _positions(u: int):
    return [i for i in range(5) if (u >> i) & 1]


def _colex(cards: np.ndarray, positions) -> np.ndarray:
    """Colex index of the sub-hand at the given positions (rows must be sorted)."""
    idx = np.zeros(len(cards), dtype=np.int64)
    for j, p in enumerate(positions):
        idx += pe._COMB[cards[:, p], j + 1]
    return idx


def _subset_sums(per_hand: np.ndarray) -> dict:
    """S[k][colex(X)] = sum of per_hand over the 5-card hands containing X."""
    hands = pe.all_hands().astype(np.int64)  # row i is the hand with hand_index i
    sums = {k: np.zeros(comb(52, k)) for k in range(6)}
    for u in range(32):
        pos = _positions(u)
        sums[len(pos)] += np.bincount(_colex(hands, pos), weights=per_hand,
                                      minlength=comb(52, len(pos)))
    return sums


def _table_for(per_category: tuple) -> np.ndarray:
    per_hand = np.asarray(per_category)[pe._tables()["category_lut"]]
    sums = _subset_sums(per_hand)
    canon = pe.canonical_states().astype(np.int64)
    v = np.empty((len(canon), 32))
    for u in range(32):
        pos = _positions(u)
        v[:, u] = sums[len(pos)][_colex(canon, pos)]
    return (v @ _MOBIUS.T) / _N_DRAWS


@lru_cache(maxsize=16)
def _cached(per_category: tuple) -> np.ndarray:
    q = _table_for(per_category)
    q.setflags(write=False)
    return q


def q_table(paytable=None) -> np.ndarray:
    """Exact Q*(state_id, canonical_mask) as a read-only (134459, 32) array.

    Indexed the same way as the Q-table in CONVENTIONS.md.
    """
    return _cached(tuple(pe.paytable_vector(paytable).tolist()))


def second_moment_table(paytable=None) -> np.ndarray:
    """Exact E[payout^2 | state, mask], for variances."""
    return _cached(tuple((pe.paytable_vector(paytable) ** 2).tolist()))


@lru_cache(maxsize=1)
def state_weights() -> np.ndarray:
    """Probability of each canonical state under a uniform deal (sums to 1)."""
    counts = np.bincount(pe._tables()["state_lut"], minlength=pe.N_CANONICAL_STATES)
    w = counts / pe.N_HANDS
    w.setflags(write=False)
    return w


def optimal_masks(paytable=None) -> np.ndarray:
    """Optimal canonical mask per state (ties go to the lowest mask)."""
    return q_table(paytable).argmax(axis=1)


def policy_values(canon_masks, paytable=None) -> np.ndarray:
    """Exact expected payout of playing ``canon_masks[s]`` in each state s."""
    q = q_table(paytable)
    return q[np.arange(len(q)), np.asarray(canon_masks)]


def expected_return(canon_masks, paytable=None, states=None) -> float:
    """Exact expected payout of a per-state policy.

    states=None averages over a uniform deal; otherwise pass the ``state_id``
    array of a split (e.g. ``load_split("eval")["state_id"]``) to average over
    those hands.
    """
    vals = policy_values(canon_masks, paytable)
    if states is None:
        return float(vals @ state_weights())
    return float(vals[np.asarray(states)].mean())


def optimal_return(paytable=None) -> float:
    return expected_return(optimal_masks(paytable), paytable)
