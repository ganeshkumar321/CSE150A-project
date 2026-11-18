"""Five-card draw poker environment (Task 1).

Shared conventions (see CONVENTIONS.md for the full list)
--------------------------------------------------------
Cards
    A card is an int in 0..51:  card = 4 * rank + suit
        rank 0..12  ->  2, 3, ..., 9, T, J, Q, K, A   (ace high; ace also low in A-2-3-4-5)
        suit 0..3   ->  h, s, d, c                   (same order as UCI: Hearts, Spades, Diamonds, Clubs)
    Sorting card ints therefore sorts by rank first, then by suit.
    A hand is a length-5 sequence of distinct card ints. Position i of a hand is
    card #(i+1) of the UCI record (S{i+1}, C{i+1}); positions are never reordered
    by the simulator.

Actions
    An action is an int mask in 0..31. Bit i (value 1 << i) refers to hand
    position i, and 1 = KEEP, 0 = discard.  31 = stand pat, 0 = discard all.

Canonicalization
    canonicalize(hand) -> (canon_hand, perm, state_id)
    canon_hand[j] is the (suit-relabelled) card of ORIGINAL position perm[j].
    A mask chosen on the canonical hand is mapped back with
    canonical_mask_to_original(mask, perm) (bit j -> bit perm[j]).

Interfaces
    policy(hand) -> mask
    simulate_draw(hand, mask, rng) -> final_hand   (np.ndarray of 5 ints)
    payout(final_hand, paytable) -> float
"""

from __future__ import annotations

import itertools
import os
from functools import lru_cache
from math import comb

import numpy as np

# --------------------------------------------------------------------------
# Card encoding
# --------------------------------------------------------------------------
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "hsdc"  # UCI suit order 1..4 = Hearts, Spades, Diamonds, Clubs
N_CARDS = 52
HAND_SIZE = 5
N_ACTIONS = 32
KEEP_ALL = 31
DISCARD_ALL = 0

TEN, JACK, ACE = 8, 9, 12  # rank indices


def make_card(rank: int, suit: int) -> int:
    return 4 * rank + suit


def card_rank(card: int) -> int:
    return card >> 2


def card_suit(card: int) -> int:
    return card & 3


def card_to_str(card: int) -> str:
    return RANK_CHARS[card >> 2] + SUIT_CHARS[card & 3]


def str_to_card(s: str) -> int:
    s = s.strip()
    return make_card(RANK_CHARS.index(s[0].upper()), SUIT_CHARS.index(s[1].lower()))


def parse_hand(s: str) -> np.ndarray:
    """'Ah Kh Qh Jh Th' -> array of card ints (positions in the given order)."""
    return np.array([str_to_card(t) for t in s.split()], dtype=np.int64)


def hand_to_str(hand) -> str:
    return " ".join(card_to_str(int(c)) for c in hand)


def uci_to_cards(suits, ranks) -> np.ndarray:
    """UCI suit (1..4) and rank (1=Ace, 2..13) arrays -> card ints (same shape)."""
    suits = np.asarray(suits, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)
    return 4 * ((ranks - 2) % 13) + (suits - 1)


def cards_to_uci(cards):
    """Inverse of uci_to_cards: returns (suits 1..4, ranks 1..13)."""
    cards = np.asarray(cards, dtype=np.int64)
    r = cards >> 2
    return (cards & 3) + 1, np.where(r == ACE, 1, r + 2)


# --------------------------------------------------------------------------
# Combinatorial (colex) index of an unordered 5-card set: 0..2,598,959
# --------------------------------------------------------------------------
N_HANDS = comb(52, 5)  # 2,598,960
_COMB = np.array([[comb(n, k) for k in range(6)] for n in range(53)], dtype=np.int64)
_COMB_L = _COMB.tolist()


def hand_index(hand) -> int:
    """Order-independent index of a 5-card set (colexicographic rank)."""
    a, b, c, d, e = sorted(int(x) for x in hand)
    C = _COMB_L
    return C[a][1] + C[b][2] + C[c][3] + C[d][4] + C[e][5]


def hand_index_batch(hands) -> np.ndarray:
    s = np.sort(np.asarray(hands, dtype=np.int64), axis=1)
    return sum(_COMB[s[:, i], i + 1] for i in range(5))


@lru_cache(maxsize=None)
def all_hands() -> np.ndarray:
    """All 2,598,960 hands as sorted rows, row i has hand_index == i."""
    flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(52), 5)),
                       dtype=np.int8, count=N_HANDS * 5).reshape(-1, 5)
    out = np.empty_like(flat)
    out[hand_index_batch(flat)] = flat
    return out


# --------------------------------------------------------------------------
# Hand evaluation
# --------------------------------------------------------------------------
UCI_CLASS_NAMES = [
    "nothing", "one_pair", "two_pair", "three_of_a_kind", "straight",
    "flush", "full_house", "four_of_a_kind", "straight_flush", "royal_flush",
]

# Payout categories refine UCI class 1 (one pair) into low pair / jacks-or-better.
CATEGORY_NAMES = [
    "nothing", "low_pair", "jacks_or_better", "two_pair", "three_of_a_kind",
    "straight", "flush", "full_house", "four_of_a_kind", "straight_flush", "royal_flush",
]
CAT = {name: i for i, name in enumerate(CATEGORY_NAMES)}

# Main paytable: 9/6 Jacks-or-Better (per 1 unit bet). Alternatives live with Task 4.
JACKS_OR_BETTER = {
    "nothing": 0, "low_pair": 0, "jacks_or_better": 1, "two_pair": 2,
    "three_of_a_kind": 3, "straight": 4, "flush": 6, "full_house": 9,
    "four_of_a_kind": 25, "straight_flush": 50, "royal_flush": 800,
}


def _classify_batch(hands):
    """Vectorized evaluator. Returns (uci_class, category) int8 arrays."""
    h = np.asarray(hands, dtype=np.int64)
    ranks = np.sort(h >> 2, axis=1)
    suits = h & 3
    n = len(h)

    flush = (suits == suits[:, :1]).all(axis=1)
    distinct = (np.diff(ranks, axis=1) != 0).all(axis=1)
    wheel = (ranks == np.array([0, 1, 2, 3, ACE])).all(axis=1)
    straight = distinct & ((ranks[:, 4] - ranks[:, 0] == 4) | wheel)
    royal = straight & flush & (ranks[:, 0] == TEN)

    counts = np.zeros((n, 13), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n), 5), ranks.ravel()), 1)
    cs = -np.sort(-counts, axis=1)
    c1, c2 = cs[:, 0], cs[:, 1]
    pair_rank = np.argmax(counts == 2, axis=1)

    uci = np.zeros(n, dtype=np.int8)
    cat = np.zeros(n, dtype=np.int8)
    one_pair = (c1 == 2) & (c2 == 1)
    rules = [  # (mask, uci class, category), lowest priority first
        (one_pair & (pair_rank < JACK), 1, CAT["low_pair"]),
        (one_pair & (pair_rank >= JACK), 1, CAT["jacks_or_better"]),
        ((c1 == 2) & (c2 == 2), 2, CAT["two_pair"]),
        ((c1 == 3) & (c2 == 1), 3, CAT["three_of_a_kind"]),
        (straight, 4, CAT["straight"]),
        (flush, 5, CAT["flush"]),
        ((c1 == 3) & (c2 == 2), 6, CAT["full_house"]),
        (c1 == 4, 7, CAT["four_of_a_kind"]),
        (straight & flush, 8, CAT["straight_flush"]),
        (royal, 9, CAT["royal_flush"]),
    ]
    for m, u, c in rules:
        uci[m] = u
        cat[m] = c
    return uci, cat


def uci_class_batch(hands) -> np.ndarray:
    return _classify_batch(hands)[0]


def hand_category_batch_direct(hands) -> np.ndarray:
    return _classify_batch(hands)[1]


def _reference_category(hand) -> int:
    """Slow, readable scalar evaluator used only to cross-check the fast paths."""
    ranks = sorted(c >> 2 for c in hand)
    suits = {c & 3 for c in hand}
    counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    flush = len(suits) == 1
    straight = len(set(ranks)) == 5 and (ranks[4] - ranks[0] == 4 or ranks == [0, 1, 2, 3, ACE])
    if straight and flush:
        return CAT["royal_flush"] if ranks[0] == TEN else CAT["straight_flush"]
    if counts[0] == 4:
        return CAT["four_of_a_kind"]
    if counts == [3, 2]:
        return CAT["full_house"]
    if flush:
        return CAT["flush"]
    if straight:
        return CAT["straight"]
    if counts[0] == 3:
        return CAT["three_of_a_kind"]
    if counts == [2, 2, 1]:
        return CAT["two_pair"]
    if counts[0] == 2:
        pr = next(r for r in ranks if ranks.count(r) == 2)
        return CAT["jacks_or_better"] if pr >= JACK else CAT["low_pair"]
    return CAT["nothing"]


CATEGORY_TO_UCI = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int8)

# --------------------------------------------------------------------------
# Lookup tables (built once, cached to disk)
# --------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
SUIT_PERMS = np.array(list(itertools.permutations(range(4))), dtype=np.int64)  # (24, 4)
N_CANONICAL_STATES = 134_459


@lru_cache(maxsize=None)
def _tables():
    """Returns dict with
    category_lut[hand_index] -> payout category
    state_lut[hand_index]    -> canonical state id
    perm_lut[hand_index]     -> index into SUIT_PERMS that canonicalizes the hand
    canon_states[state_id]   -> sorted canonical hand (5 ints)
    """
    path = os.path.join(_CACHE_DIR, "tables_v1.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    hands = all_hands().astype(np.int64)
    category_lut = hand_category_batch_direct(hands)

    base = np.array([52 ** 4, 52 ** 3, 52 ** 2, 52, 1], dtype=np.int64)
    rank_part = hands & ~3
    suit_part = hands & 3
    best_key = np.full(N_HANDS, np.iinfo(np.int64).max)
    best_perm = np.zeros(N_HANDS, dtype=np.int64)
    for p_idx, p in enumerate(SUIT_PERMS):
        relabeled = np.sort(rank_part | p[suit_part], axis=1)
        key = relabeled @ base
        better = key < best_key
        best_key[better] = key[better]
        best_perm[better] = p_idx
    keys, state_lut = np.unique(best_key, return_inverse=True)
    canon_states = np.stack([(keys // b) % 52 for b in base], axis=1)

    out = dict(category_lut=category_lut.astype(np.int8),
               state_lut=state_lut.astype(np.int32),
               perm_lut=best_perm.astype(np.int8),
               canon_states=canon_states.astype(np.int8))
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def hand_category(hand) -> int:
    """Payout category index (see CATEGORY_NAMES) of a 5-card hand."""
    return int(_tables()["category_lut"][hand_index(hand)])


def hand_category_batch(hands) -> np.ndarray:
    return _tables()["category_lut"][hand_index_batch(hands)]


def uci_class(hand) -> int:
    """UCI CLASS label (0..9) of a 5-card hand."""
    return int(CATEGORY_TO_UCI[hand_category(hand)])


# --------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------
def paytable_vector(paytable=None) -> np.ndarray:
    paytable = JACKS_OR_BETTER if paytable is None else paytable
    return np.array([float(paytable[name]) for name in CATEGORY_NAMES])


def payout(final_hand, paytable=None) -> float:
    """Reward for a final hand under a paytable (dict category name -> payout)."""
    paytable = JACKS_OR_BETTER if paytable is None else paytable
    return float(paytable[CATEGORY_NAMES[hand_category(final_hand)]])


def payout_batch(final_hands, paytable=None) -> np.ndarray:
    return paytable_vector(paytable)[hand_category_batch(final_hands)]


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------
def canonicalize(hand):
    """Map a hand to its canonical representative.

    Canonical form: relabel suits by the permutation (out of 24) that makes the
    rank-then-suit sorted card tuple lexicographically smallest, then sort.

    Returns (canon_hand, perm, state_id):
        canon_hand : tuple of 5 sorted card ints
        perm       : tuple, perm[j] = original position of canonical card j
        state_id   : int in 0..134,458 (index into canonical_states())
    """
    t = _tables()
    idx = hand_index(hand)
    p = SUIT_PERMS[t["perm_lut"][idx]]
    relabeled = [(int(c) & ~3) | int(p[int(c) & 3]) for c in hand]
    perm = tuple(sorted(range(5), key=relabeled.__getitem__))
    canon = tuple(relabeled[i] for i in perm)
    return canon, perm, int(t["state_lut"][idx])


def state_id(hand) -> int:
    return int(_tables()["state_lut"][hand_index(hand)])


def state_id_batch(hands) -> np.ndarray:
    return _tables()["state_lut"][hand_index_batch(hands)]


def canonical_states() -> np.ndarray:
    """(134459, 5) array; row s is the canonical hand of state s."""
    return _tables()["canon_states"]


def canonical_mask_to_original(mask: int, perm) -> int:
    """Bit j of a canonical-hand mask -> bit perm[j] of the original hand mask."""
    out = 0
    for j in range(5):
        if (mask >> j) & 1:
            out |= 1 << perm[j]
    return out


def original_mask_to_canonical(mask: int, perm) -> int:
    out = 0
    for j in range(5):
        if (mask >> perm[j]) & 1:
            out |= 1 << j
    return out


def canonical_policy(canon_policy):
    """Wrap a policy defined on canonical hands into a policy on raw hands.

    canon_policy(canon_hand, state_id) -> canonical mask
    """
    def policy(hand):
        canon, perm, sid = canonicalize(hand)
        return canonical_mask_to_original(canon_policy(canon, sid), perm)
    return policy


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def kept_positions(mask: int):
    return [i for i in range(5) if (mask >> i) & 1]


def discarded_positions(mask: int):
    return [i for i in range(5) if not (mask >> i) & 1]


def mask_from_keep(positions) -> int:
    m = 0
    for i in positions:
        m |= 1 << i
    return m


def mask_to_str(mask: int) -> str:
    """Human-readable mask, position 0 first, e.g. 'KKDDK'."""
    return "".join("K" if (mask >> i) & 1 else "D" for i in range(5))


# --------------------------------------------------------------------------
# Draw simulator
# --------------------------------------------------------------------------
def remaining_deck(hand) -> np.ndarray:
    """The 47 cards not in the original hand (sorted)."""
    m = np.ones(N_CARDS, dtype=bool)
    m[np.asarray(hand, dtype=np.int64)] = False
    return np.flatnonzero(m)


def simulate_draw(hand, mask: int, rng: np.random.Generator) -> np.ndarray:
    """Keep the masked cards in place; fill each discarded position with a card
    drawn uniformly without replacement from the 47 cards outside the ORIGINAL
    hand (discards never return to the deck). Positions are preserved."""
    out = np.array(hand, dtype=np.int64)
    holes = [i for i in range(5) if not (mask >> i) & 1]
    if not holes:
        return out
    deck = remaining_deck(out)
    u = rng.random(len(holes))
    for k, pos in enumerate(holes):  # partial Fisher-Yates
        j = k + int(u[k] * (47 - k))
        deck[k], deck[j] = deck[j], deck[k]
        out[pos] = deck[k]
    return out


def simulate_draw_batch(hands, masks, rng: np.random.Generator) -> np.ndarray:
    """Vectorized simulate_draw for N hands/masks at once."""
    hands = np.asarray(hands, dtype=np.int64)
    masks = np.broadcast_to(np.asarray(masks, dtype=np.int64), (len(hands),))
    n = len(hands)
    keys = rng.random((n, N_CARDS))
    keys[np.arange(n)[:, None], hands] = np.inf  # never draw original cards
    draws = np.argpartition(keys, 5, axis=1)[:, :5]  # 5 distinct random cards each
    draws = np.take_along_axis(draws, np.argsort(np.take_along_axis(keys, draws, axis=1), axis=1), axis=1)
    keep = ((masks[:, None] >> np.arange(5)) & 1).astype(bool)
    slot = np.cumsum(~keep, axis=1) - 1  # k-th hole takes k-th draw
    filled = np.take_along_axis(draws, np.clip(slot, 0, 4), axis=1)
    return np.where(keep, hands, filled)


def play(policy, hand, rng, paytable=None) -> float:
    """One episode: choose mask, draw, return payout."""
    return payout(simulate_draw(hand, policy(hand), rng), paytable)


# --------------------------------------------------------------------------
# Exact expected values (ground truth for testing; not used by the agent)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _combo_idx(k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(47), k)), dtype=np.int64).reshape(-1, k)


def exact_action_value(hand, mask: int, paytable=None) -> float:
    """E[payout | hand, mask], by enumerating every possible draw."""
    hand = np.asarray(hand, dtype=np.int64)
    kept = hand[[i for i in range(5) if (mask >> i) & 1]]
    k = 5 - len(kept)
    if k == 0:
        return payout(hand, paytable)
    draws = remaining_deck(hand)[_combo_idx(k)]
    finals = np.concatenate([np.broadcast_to(kept, (len(draws), len(kept))), draws], axis=1)
    return float(payout_batch(finals, paytable).mean())


def exact_action_values(hand, paytable=None) -> np.ndarray:
    """Exact Q*(hand, a) for all 32 masks (a few seconds per hand)."""
    return np.array([exact_action_value(hand, a, paytable) for a in range(N_ACTIONS)])
