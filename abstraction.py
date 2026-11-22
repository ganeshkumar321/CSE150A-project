"""State (and action) abstractions for the tabular agent (Task 4).

Every abstraction is defined on top of the canonical states from Task 1, as two
lookup tables:

    state_map[state_id]       -> abstract state z in 0..n_states-1
    action_map[state_id, a]   -> canonical mask (0..31) that abstract action a means

So a Q-table for any abstraction has shape ``(n_states, n_actions)`` and the
agent only ever does array lookups. Three abstractions are provided:

``canonical``  134,459 states x 32 masks. The full state from Task 1 (identity maps).
``ranks``      6,175 states x 32 masks. Only the sorted ranks of the canonical
               hand; suits are ignored, so the agent cannot see flush draws.
``coarse``     A few hundred states x 11 hold rules. The state is a feature
               tuple (made hand, flush/straight/straight-flush/royal draw sizes,
               number of high cards) and each action is a rule such as "keep
               the pairs" or "keep the flush draw", turned into a concrete mask
               for each hand. Positional masks would mean different things in
               hands that share a feature tuple, so a coarse state needs
               semantic actions.

Usage
-----
    ab = get_abstraction("coarse")
    z, sid, perm = ab.encode(hand)            # abstract state for the Q-table
    mask = ab.to_mask(a, sid, perm)           # abstract action -> real mask
    final = pe.simulate_draw(hand, mask, rng)

    # batched (training / evaluation)
    z, sid, perm = ab.encode_batch(hands)
    masks = ab.to_mask_batch(actions, sid, perm)

The tables are built once (about 10 s for ``coarse``) and cached in
``data/cache/``, which is git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import exact_values as ev
import poker_env as pe

_CACHE_PATH = os.path.join(pe._CACHE_DIR, "abstractions_v1.npz")

# --------------------------------------------------------------------------
# Hold rules (abstract actions of the coarse abstraction)
# --------------------------------------------------------------------------
HOLD_RULES = [
    "draw_five",     # discard everything
    "stand",         # keep everything
    "pairs",         # keep every card whose rank appears 2+ times (pair, two pair, trips, ...)
    "flush_draw",    # keep all cards of the most common suit
    "straight_draw", # keep one card per rank in the 5-rank window with the most ranks
    "sf_draw",       # keep the suited cards in the best 5-rank window
    "royal_draw",    # keep the suited T-J-Q-K-A cards of the best suit
    "high_all",      # keep one card of every rank J or higher
    "high_suited",   # keep the J+ cards of the suit that has the most of them
    "high_top1",     # keep the single highest card if it is J or higher
    "high_top2",     # keep the two highest J+ ranks
]
STRAIGHT_LEVELS = ["<=2", "3", "4_inside", "4_open", "5"]
_WINDOWS = list(range(-1, 9))  # window w covers ranks w..w+4; w = -1 is A-2-3-4-5


def _pick(groups):
    """Largest group of positions; ties go to the group with higher ranks (last wins)."""
    best = []
    for g in groups:
        if len(g) >= len(best):
            best = g
    return best


def _coarse_hand(cards):
    """Feature tuple and the 11 hold-rule position lists for one canonical hand."""
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    count = {r: ranks.count(r) for r in ranks}

    def window_ranks(w):  # ranks of the window, with the ace counted as -1 for the wheel
        return [(r if r >= 0 else pe.ACE) for r in range(w, w + 5)]

    def first_pos(r, suit=None):
        for i in range(5):
            if ranks[i] == r and (suit is None or suits[i] == suit):
                return i
        return None

    # Suits are scanned high-to-low by their best card so ties prefer higher cards.
    suit_order = sorted(set(suits), key=lambda s: max(r for r, t in zip(ranks, suits) if t == s))
    by_suit = {s: [i for i in range(5) if suits[i] == s] for s in suit_order}

    straight_groups = []
    for w in _WINDOWS:
        straight_groups.append([p for r in window_ranks(w) if (p := first_pos(r)) is not None])
    sf_groups = [[p for r in window_ranks(w) if (p := first_pos(r, s)) is not None]
                 for s in suit_order for w in _WINDOWS]
    royal_groups = [[i for i in by_suit[s] if ranks[i] >= pe.TEN] for s in suit_order]
    high_ranks = sorted({r for r in ranks if r >= pe.JACK})
    high_suited_groups = [[i for i in by_suit[s] if ranks[i] >= pe.JACK] for s in suit_order]

    holds = {
        "draw_five": [],
        "stand": list(range(5)),
        "pairs": [i for i in range(5) if count[ranks[i]] >= 2],
        "flush_draw": _pick(by_suit.values()),
        "straight_draw": _pick(straight_groups),
        "sf_draw": _pick(sf_groups),
        "royal_draw": _pick(royal_groups),
        "high_all": [first_pos(r) for r in high_ranks],
        "high_suited": _pick(high_suited_groups),
        "high_top1": [first_pos(r) for r in high_ranks[-1:]],
        "high_top2": [first_pos(r) for r in high_ranks[-2:]],
    }

    straight_n = len(holds["straight_draw"])
    if straight_n == 4:
        held = sorted(ranks[i] for i in holds["straight_draw"])
        consecutive = held[3] - held[0] == 3
        open_ended = consecutive and pe.ACE not in held
        level = "4_open" if open_ended else "4_inside"
    else:
        level = {5: "5", 3: "3"}.get(straight_n, "<=2")

    features = (
        pe.hand_category(cards),                 # made hand, 0..10
        len(holds["flush_draw"]),                # cards of the most common suit, 2..5
        STRAIGHT_LEVELS.index(level),            # straight draw, 0..4
        len(holds["sf_draw"]),                   # straight-flush draw, 1..5
        len(holds["royal_draw"]),                # royal draw, 0..5
        len(high_ranks),                         # distinct J+ ranks, 0..4
    )
    masks = [pe.mask_from_keep(holds[name]) for name in HOLD_RULES]
    return features, masks


COARSE_FEATURES = ["made", "flush_n", "straight", "sf_n", "royal_n", "high_n"]


def _build_tables():
    canon = pe.canonical_states().astype(np.int64)
    n = len(canon)

    # ranks: sorted ranks of the canonical hand (a multiset of 5 ranks)
    rank_keys = (canon >> 2) @ (13 ** np.arange(4, -1, -1))
    _, ranks_state = np.unique(rank_keys, return_inverse=True)

    # coarse: feature tuple + hold-rule masks
    feats = np.empty((n, len(COARSE_FEATURES)), dtype=np.int8)
    coarse_actions = np.empty((n, len(HOLD_RULES)), dtype=np.int8)
    for s in range(n):
        f, m = _coarse_hand(canon[s].tolist())
        feats[s] = f
        coarse_actions[s] = m
    coarse_keys, coarse_state = np.unique(feats, axis=0, return_inverse=True)

    return dict(ranks_state=ranks_state.astype(np.int32),
                coarse_state=coarse_state.astype(np.int32).ravel(),
                coarse_actions=coarse_actions,
                coarse_keys=coarse_keys.astype(np.int8))


@lru_cache(maxsize=None)
def _tables():
    if os.path.exists(_CACHE_PATH):
        with np.load(_CACHE_PATH) as z:
            return {k: z[k] for k in z.files}
    out = _build_tables()
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    np.savez_compressed(_CACHE_PATH, **out)
    return out


# --------------------------------------------------------------------------
# Abstraction object
# --------------------------------------------------------------------------
def canonical_batch(hands):
    """Vectorized pe.canonicalize: returns (state_id (N,), perm (N, 5))."""
    hands = np.asarray(hands, dtype=np.int64)
    t = pe._tables()
    idx = pe.hand_index_batch(hands)
    p = pe.SUIT_PERMS[t["perm_lut"][idx].astype(np.int64)]
    relabeled = (hands & ~3) | np.take_along_axis(p, hands & 3, axis=1)
    perm = np.argsort(relabeled, axis=1)
    return t["state_lut"][idx].astype(np.int64), perm


@dataclass(frozen=True)
class Abstraction:
    name: str
    state_map: np.ndarray    # (134459,) abstract state of each canonical state
    action_map: np.ndarray   # (134459, n_actions) canonical mask of each abstract action
    action_names: tuple

    @property
    def n_states(self) -> int:
        return int(self.state_map.max()) + 1

    @property
    def n_actions(self) -> int:
        return self.action_map.shape[1]

    # ---- single hand ----
    def encode(self, hand):
        """hand -> (abstract state, canonical state_id, perm)."""
        _, perm, sid = pe.canonicalize(hand)
        return int(self.state_map[sid]), sid, perm

    def to_mask(self, action: int, sid: int, perm) -> int:
        """Abstract action -> mask over the ORIGINAL hand's positions."""
        return pe.canonical_mask_to_original(int(self.action_map[sid, action]), perm)

    def state_encoder(self):
        """Encoder for Task 2's ``QLearningAgent(encoder=..., n_states=ab.n_states, state_encoding=ab.name)``.

        Only for abstractions whose actions are the 32 canonical-position masks
        (canonical, ranks); the coarse abstraction's hold rules need its own action set.
        """
        if self.n_actions != 32 or not (self.action_map == np.arange(32)).all():
            raise ValueError(f"{self.name!r} does not use the 32 canonical masks")

        def encode(hand):
            z, _, perm = self.encode(hand)
            return z, perm
        return encode

    def policy(self, q_table):
        """Greedy policy(hand) -> mask from a Q-table of shape (n_states, n_actions)."""
        def act(hand):
            z, sid, perm = self.encode(hand)
            return self.to_mask(int(np.argmax(q_table[z])), sid, perm)
        return act

    # ---- batches ----
    def encode_batch(self, hands):
        sid, perm = canonical_batch(hands)
        return self.state_map[sid], sid, perm

    def to_mask_batch(self, actions, sid, perm) -> np.ndarray:
        cm = self.action_map[sid, actions].astype(np.int64)
        bits = (cm[:, None] >> np.arange(5)) & 1
        return (bits << perm).sum(axis=1)

    # ---- exact analysis (uses exact_values) ----
    def greedy_canonical_masks(self, q_table) -> np.ndarray:
        """Canonical mask the greedy policy plays in each of the 134,459 states."""
        best = np.asarray(q_table)[self.state_map].argmax(axis=1)
        return self.action_map[np.arange(len(best)), best].astype(np.int64)

    def exact_q(self, paytable=None) -> np.ndarray:
        """True Q(z, a): exact expected payout averaged over the hands in z (uniform deal)."""
        q = ev.q_table(paytable)
        w = ev.state_weights()
        vals = q[np.arange(len(q))[:, None], self.action_map]
        num = np.stack([np.bincount(self.state_map, weights=w * vals[:, a], minlength=self.n_states)
                        for a in range(self.n_actions)], axis=1)
        den = np.bincount(self.state_map, weights=w, minlength=self.n_states)
        return num / den[:, None]

    def best_canonical_masks(self, paytable=None) -> np.ndarray:
        """The best policy this abstraction can express, as a canonical mask per state."""
        return self.greedy_canonical_masks(self.exact_q(paytable))


@lru_cache(maxsize=None)
def get_abstraction(name: str) -> Abstraction:
    n = pe.N_CANONICAL_STATES
    masks32 = np.broadcast_to(np.arange(32, dtype=np.int8), (n, 32))
    mask_names = tuple(pe.mask_to_str(m) for m in range(32))
    if name == "canonical":
        return Abstraction(name, np.arange(n, dtype=np.int32), masks32, mask_names)
    t = _tables()
    if name == "ranks":
        return Abstraction(name, t["ranks_state"], masks32, mask_names)
    if name == "coarse":
        return Abstraction(name, t["coarse_state"], t["coarse_actions"], tuple(HOLD_RULES))
    raise KeyError(f"unknown abstraction {name!r}; choose from {ABSTRACTIONS}")


ABSTRACTIONS = ("canonical", "ranks", "coarse")


def coarse_state_features(z: int) -> dict:
    """Feature values that define coarse state z (for reading results)."""
    f = _tables()["coarse_keys"][z]
    out = dict(zip(COARSE_FEATURES, (int(x) for x in f)))
    out["made"] = pe.CATEGORY_NAMES[out["made"]]
    out["straight"] = STRAIGHT_LEVELS[out["straight"]]
    return out
