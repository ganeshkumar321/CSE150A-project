"""Baseline strategies (Task 3).

Every policy follows the shared interface ``policy(hand) -> mask``: ``hand`` is
5 card ints in dealt order, and the mask (0..31) has bit i = 1 to KEEP
position i (see CONVENTIONS.md). Masks always refer to the original positions.

Policies, from weakest to strongest:

    random_policy(seed)  uniform random mask (reference)
    keep_all             stand pat (reference)
    pairs_policy         keep any made hand, else discard everything
    basic_policy         made hands > 4-card flush > straight draws > discard all
    strategy_policy      16-rule priority list (STRATEGY_RULES) for 9/6 Jacks-or-Better
    oracle_policy        argmax of exact expected payout (upper bound; slow)

The heuristics look only at ranks and suits, so they are deterministic and give
the same decision on every hand in a canonical state (up to which of two
interchangeable cards is kept). They are tuned for the main paytable
``pe.JACKS_OR_BETTER``; ``oracle_policy`` accepts any paytable.

``explain(hand, rules)`` returns ``(mask, rule_name)`` so a decision can be traced.
Run ``python baselines.py`` to print the representative-hand table used in the
report.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import poker_env as pe

HIGH = 9            # rank index of J; J, Q, K, A are "high cards" (a pair of them pays)
ROYAL_LOW = 8       # rank index of T; T..A are the royal ranks
ACE = 12


# --------------------------------------------------------------------------
# Hand features. Each returns hand positions (a list) or None if absent.
# --------------------------------------------------------------------------
def _ranks(hand):
    return [int(c) >> 2 for c in hand]


def _suits(hand):
    return [int(c) & 3 for c in hand]


def _by_rank(hand):
    groups = {}
    for i, r in enumerate(_ranks(hand)):
        groups.setdefault(r, []).append(i)
    return groups


def _by_suit(hand):
    groups = {}
    for i, s in enumerate(_suits(hand)):
        groups.setdefault(s, []).append(i)
    return groups


def _windows():
    """All 10 straight windows as rank sets; the wheel uses the ace (12) as low."""
    yield frozenset({ACE, 0, 1, 2, 3})
    for low in range(0, 9):
        yield frozenset(range(low, low + 5))


def _one_per_rank(hand, positions, ranks):
    """Pick one position for each rank in ``ranks`` (first occurrence)."""
    r = _ranks(hand)
    chosen = {}
    for i in positions:
        if r[i] in ranks and r[i] not in chosen:
            chosen[r[i]] = i
    return sorted(chosen.values())


def _is_flush(hand):
    return len(set(_suits(hand))) == 1


def _is_straight(hand):
    rs = set(_ranks(hand))
    return len(rs) == 5 and any(rs == w for w in _windows())


def made_straight_flush(hand):
    """Royal or straight flush: keep all five."""
    return list(range(5)) if _is_flush(hand) and _is_straight(hand) else None


def four_of_a_kind(hand):
    for pos in _by_rank(hand).values():
        if len(pos) == 4:
            return pos
    return None


def full_house(hand):
    sizes = sorted(len(p) for p in _by_rank(hand).values())
    return list(range(5)) if sizes == [2, 3] else None


def flush(hand):
    return list(range(5)) if _is_flush(hand) else None


def straight(hand):
    return list(range(5)) if _is_straight(hand) else None


def three_of_a_kind(hand):
    for pos in _by_rank(hand).values():
        if len(pos) == 3:
            return pos
    return None


def two_pair(hand):
    pairs = [p for p in _by_rank(hand).values() if len(p) == 2]
    return sorted(pairs[0] + pairs[1]) if len(pairs) == 2 else None


def high_pair(hand):
    for r, pos in _by_rank(hand).items():
        if len(pos) == 2 and r >= HIGH:
            return pos
    return None


def low_pair(hand):
    for r, pos in _by_rank(hand).items():
        if len(pos) == 2 and r < HIGH:
            return pos
    return None


def any_made_hand(hand):
    """Anything that already scores a category above 'nothing' (incl. low pairs)."""
    for rule in (made_straight_flush, flush, straight, full_house):
        if rule(hand):
            return list(range(5))
    for rule in (four_of_a_kind, three_of_a_kind, two_pair, high_pair, low_pair):
        pos = rule(hand)
        if pos:
            return pos
    return None


def _suited_draw(hand, n, allowed_windows):
    """``n`` cards of one suit whose ranks all lie in one of ``allowed_windows``."""
    r = _ranks(hand)
    for pos in _by_suit(hand).values():
        if len(pos) < n:
            continue
        for w in allowed_windows:
            inside = [i for i in pos if r[i] in w]
            if len(inside) >= n:
                return sorted(inside[:n])
    return None


ROYAL_WINDOW = frozenset(range(ROYAL_LOW, ACE + 1))


def four_to_royal(hand):
    return _suited_draw(hand, 4, [ROYAL_WINDOW])


def three_to_royal(hand):
    return _suited_draw(hand, 3, [ROYAL_WINDOW])


def four_to_straight_flush(hand):
    return _suited_draw(hand, 4, list(_windows()))


def four_to_flush(hand):
    for pos in _by_suit(hand).values():
        if len(pos) == 4:
            return pos
    return None


def four_to_open_straight(hand):
    """Four consecutive ranks that complete at either end (2345 up to TJQK).
    A-2-3-4 and J-Q-K-A complete at one end only, so they count as inside draws."""
    rs = set(_ranks(hand))
    for low in range(0, 9):  # 2345 .. TJQK
        need = set(range(low, low + 4))
        if need <= rs:
            return _one_per_rank(hand, range(5), need)
    return None


def four_to_inside_straight(hand):
    """Four distinct ranks inside one straight window (one rank missing)."""
    rs = set(_ranks(hand))
    for w in _windows():
        hit = rs & w
        if len(hit) == 4:
            return _one_per_rank(hand, range(5), hit)
    return None


def high_cards(hand):
    """Keep at most two high cards (J+). Prefer two suited high cards; otherwise
    keep the two lowest, which leave the most room for a straight."""
    r, s = _ranks(hand), _suits(hand)
    hi = [i for i in range(5) if r[i] >= HIGH]
    if not hi:
        return None
    suited = [(i, j) for a, i in enumerate(hi) for j in hi[a + 1:] if s[i] == s[j]]
    if suited:
        return sorted(min(suited, key=lambda p: r[p[0]] + r[p[1]]))
    return sorted(sorted(hi, key=lambda i: r[i])[:2])


# --------------------------------------------------------------------------
# Rule lists. The first rule that applies decides; nothing applies -> discard all.
# --------------------------------------------------------------------------
PAIRS_RULES = [
    ("made hand", any_made_hand),
]

BASIC_RULES = [
    ("made hand", any_made_hand),
    ("4 to a flush", four_to_flush),
    ("4 to an open straight", four_to_open_straight),
    ("4 to an inside straight", four_to_inside_straight),
]

# Ordered by expected payout under 9/6 Jacks-or-Better (exact EV of each holding
# in brackets, for a typical hand; see report/task3_baselines.md). Rules that can
# never apply to the same hand (e.g. flush vs. three of a kind) are ordered by EV
# anyway so that the list reads top-down.
STRATEGY_RULES = [
    ("royal / straight flush", made_straight_flush),      # 800 / 50
    ("four of a kind", four_of_a_kind),                   # 25
    ("full house", full_house),                           # 9
    ("4 to a royal flush", four_to_royal),                # ~18.4; breaks a made flush, straight or high pair
    ("flush", flush),                                     # 6
    ("three of a kind", three_of_a_kind),                 # ~4.30
    ("straight", straight),                               # 4
    ("4 to a straight flush", four_to_straight_flush),    # ~2.1-3.6; beats a high pair
    ("two pair", two_pair),                               # ~2.60
    ("high pair", high_pair),                             # ~1.54
    ("3 to a royal flush", three_to_royal),               # ~1.3-1.4
    ("4 to a flush", four_to_flush),                      # ~1.2-1.3; beats a low pair
    ("low pair", low_pair),                               # ~0.82
    ("4 to an open straight", four_to_open_straight),     # ~0.68-0.87
    ("high cards", high_cards),                           # ~0.45-0.6
    # inside straights without high cards (~0.34) are worse than discarding all (~0.36)
]


def explain(hand, rules):
    """Return (mask, name of the rule that fired) for a hand under a rule list."""
    for name, rule in rules:
        pos = rule(hand)
        if pos:
            return pe.mask_from_keep(pos), name
    return pe.DISCARD_ALL, "discard all"


def make_rule_policy(rules):
    def policy(hand):
        return explain(hand, rules)[0]
    return policy


pairs_policy = make_rule_policy(PAIRS_RULES)
basic_policy = make_rule_policy(BASIC_RULES)
strategy_policy = make_rule_policy(STRATEGY_RULES)


def keep_all(hand):
    return pe.KEEP_ALL


def random_policy(seed=0):
    """Uniform random mask. Returns a policy with its own generator, so pass a
    seed for reproducibility (the mask stream is independent of the draw rng)."""
    rng = np.random.default_rng(seed)

    def policy(hand):
        return int(rng.integers(pe.N_ACTIONS))
    return policy


# --------------------------------------------------------------------------
# Oracle: exact optimum (upper bound), cached per canonical state
# --------------------------------------------------------------------------
_oracle_cache = {}


def _paytable_key(paytable):
    return tuple(pe.paytable_vector(paytable))


def canonical_best(state_id, canon_hand, paytable=None):
    """(best canonical mask, exact action values) for one canonical state."""
    key = (_paytable_key(paytable), state_id)
    if key not in _oracle_cache:
        q = pe.exact_action_values(canon_hand, paytable)
        _oracle_cache[key] = (int(np.argmax(q)), q)
    return _oracle_cache[key]


def oracle_policy(paytable=None):
    """Optimal policy by exact enumeration (~0.1 s per new canonical state)."""
    def canon(canon_hand, sid):
        return canonical_best(sid, np.array(canon_hand), paytable)[0]
    return pe.canonical_policy(canon)


def regret(hand, mask, paytable=None):
    """Exact EV lost by playing ``mask`` instead of the optimal action."""
    canon, perm, sid = pe.canonicalize(hand)
    best, q = canonical_best(sid, np.array(canon), paytable)
    return float(q[best] - q[pe.original_mask_to_canonical(mask, perm)])


POLICIES = {
    "random": random_policy,
    "keep all": lambda: keep_all,
    "pairs": lambda: pairs_policy,
    "basic": lambda: basic_policy,
    "strategy": lambda: strategy_policy,
    "oracle": oracle_policy,
}


# --------------------------------------------------------------------------
# Representative hands (documented in the report)
# --------------------------------------------------------------------------
REPRESENTATIVE_HANDS = [
    # (hand, what it tests)
    ("Th Jh Qh Kh Ah", "made royal flush"),
    ("9s 9h 9d 9c 4h", "four of a kind"),
    ("Kh Ks Kd 4c 4h", "full house"),
    ("Th Jh Qh Kh 3h", "flush containing 4 to a royal: break it"),
    ("Th Jh Qh Kh 9s", "straight containing 4 to a royal: break it"),
    ("Jh Qh Kh Ah Jd", "high pair vs 4 to a royal"),
    ("2h 3h 5h 9h Kh", "made flush"),
    ("7h 7s 7d Kc 2h", "three of a kind"),
    ("5h 6s 7d 8c 9h", "made straight"),
    ("8h 9h Th Jh Jd", "high pair vs open 4 to a straight flush"),
    ("8h 8s 4d 4c Kh", "two pair"),
    ("Qh Qs 2d 7c 9h", "high pair"),
    ("Qh Qs 3s 7s 9s", "high pair vs 4 to a flush"),
    ("Th Qh Ah 4c 4d", "low pair vs 3 to a royal"),
    ("Kh Qh 7h 2s 7d", "low pair vs 3 to a flush with high cards"),
    ("3h 5h 9h Kh 5d", "low pair vs 4 to a flush"),
    ("6h 6s 7d 8c 9h", "low pair vs open straight"),
    ("5h 6s 7d 8c Kh", "open straight vs high card"),
    ("2h 7s 8d Tc Jh", "inside straight vs high card"),
    ("2h 3s 5d 6c 9h", "inside straight, no high cards"),
    ("Jh Qs 4d 7c 2h", "two unsuited high cards"),
    ("Jh Qh As 4c 7d", "suited high cards vs three high cards"),
    ("2h 4s 7d 8c Th", "nothing: discard all"),
]


def representative_table(paytable=None):
    rows = []
    for text, note in REPRESENTATIVE_HANDS:
        hand = pe.parse_hand(text)
        canon, perm, sid = pe.canonicalize(hand)
        best_c, q = canonical_best(sid, np.array(canon), paytable)
        best = pe.canonical_mask_to_original(best_c, perm)
        row = {"hand": text, "tests": note}
        for name, rules in [("basic", BASIC_RULES), ("strategy", STRATEGY_RULES)]:
            mask, rule = explain(hand, rules)
            row[name] = (mask, rule, q[pe.original_mask_to_canonical(mask, perm)])
        row["optimal"] = (best, None, q[best_c])
        rows.append(row)
    return rows


def _held(hand, mask):
    kept = [pe.card_to_str(int(hand[i])) for i in pe.kept_positions(mask)]
    return " ".join(kept) if kept else "(none)"


def representative_markdown(paytable=None):
    lines = ["| Hand | Tests | Basic holds (EV) | Strategy holds (rule, EV) | Optimal holds (EV) |",
             "|---|---|---|---|---|"]
    for row in representative_table(paytable):
        hand = pe.parse_hand(row["hand"])
        bm, _, bev = row["basic"]
        sm, srule, sev = row["strategy"]
        om, _, oev = row["optimal"]
        flag = "" if abs(sev - oev) < 1e-12 else " ✗"
        lines.append(f"| {row['hand']} | {row['tests']} | {_held(hand, bm)} ({bev:.3f}) | "
                     f"{_held(hand, sm)} ({srule}, {sev:.3f}){flag} | {_held(hand, om)} ({oev:.3f}) |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(representative_markdown())
