"""Unit tests for baselines. Run with `python test_baselines.py` (or pytest)."""

import numpy as np

import baselines as bl
import poker_env as pe


def held(text, policy=bl.strategy_policy):
    hand = pe.parse_hand(text)
    return sorted(pe.card_to_str(int(hand[i])) for i in pe.kept_positions(policy(hand)))


def cards(text):
    return sorted(text.split())


def test_reference_policies():
    h = pe.parse_hand("2h 4s 7d 8c Th")
    assert bl.keep_all(h) == pe.KEEP_ALL
    a, b = bl.random_policy(7), bl.random_policy(7)
    ma = [a(h) for _ in range(2000)]
    assert ma == [b(h) for _ in range(2000)]  # seeded -> reproducible
    counts = np.bincount(ma, minlength=32)
    assert counts.min() > 0 and set(ma) <= set(range(32))


def test_strategy_rule_priorities():
    cases = {
        "Th Jh Qh Kh Ah": ("royal / straight flush", "Th Jh Qh Kh Ah"),
        "Th Jh Qh Kh 3h": ("4 to a royal flush", "Th Jh Qh Kh"),       # breaks flush
        "Th Jh Qh Kh 9s": ("4 to a royal flush", "Th Jh Qh Kh"),       # breaks straight
        "Jh Qh Kh Ah Jd": ("4 to a royal flush", "Jh Qh Kh Ah"),       # breaks high pair
        "8h 9h Th Jh Jd": ("4 to a straight flush", "8h 9h Th Jh"),    # breaks high pair
        "Qh Qs 3s 7s 9s": ("high pair", "Qh Qs"),                      # pair over 4-flush
        "Th Qh Ah 4c 4d": ("3 to a royal flush", "Th Qh Ah"),          # 3RF over low pair
        "3h 5h 9h Kh 5d": ("4 to a flush", "3h 5h 9h Kh"),             # 4-flush over low pair
        "6h 6s 7d 8c 9h": ("low pair", "6h 6s"),                       # low pair over open straight
        "5h 6s 7d 8c Kh": ("4 to an open straight", "5h 6s 7d 8c"),
        "Jh Qh As 4c 7d": ("high cards", "Jh Qh"),                     # suited high cards
        "Jd Qs Ac 4c 7d": ("high cards", "Jd Qs"),                     # two lowest high cards
        "2h 3s 5d 6c 9h": ("discard all", ""),                         # inside straight not kept
    }
    for text, (rule, keep) in cases.items():
        mask, got = bl.explain(pe.parse_hand(text), bl.STRATEGY_RULES)
        assert got == rule, (text, got)
        assert held(text) == cards(keep), (text, held(text))


def test_draw_features():
    f = lambda fn, s: fn(pe.parse_hand(s))
    assert f(bl.four_to_open_straight, "2h 3s 4d 5c 9h")          # 2345 is open (A or 6)
    assert not f(bl.four_to_open_straight, "Jh Qs Kd Ac 3h")      # JQKA is one-sided
    assert not f(bl.four_to_open_straight, "Ah 2s 3d 4c 9h")      # A234 is one-sided
    assert f(bl.four_to_inside_straight, "Ah 2s 3d 4c 9h")
    assert f(bl.four_to_inside_straight, "Jh Qs Kd Ac 3h")
    assert f(bl.four_to_straight_flush, "Ah 2h 3h 5h 9s")         # wheel straight-flush draw
    assert not f(bl.four_to_straight_flush, "2h 3h 4h 9h Ks")
    assert len(f(bl.four_to_open_straight, "6h 6s 7d 8c 9h")) == 4  # one card per rank


def test_basic_and_pairs_policies():
    assert held("Qh Qs 3s 7s 9s", bl.basic_policy) == cards("Qh Qs")
    assert held("3h 5h 9h Kh 5d", bl.basic_policy) == cards("5h 5d")        # pairs first
    assert held("2h 3s 5d 6c 9h", bl.basic_policy) == cards("2h 3s 5d 6c")  # keeps inside draws
    assert held("2h 3h 5h 9h Ks", bl.pairs_policy) == []
    assert held("5h 6s 7d 8c 9h", bl.pairs_policy) == cards("5h 6s 7d 8c 9h")


def test_invariance_under_reorder_and_suit_relabel():
    rng = np.random.default_rng(0)
    deck = np.arange(52)
    for _ in range(300):
        h = rng.choice(deck, 5, replace=False)
        order = rng.permutation(5)
        sigma = rng.permutation(4)
        h2 = np.array([4 * (c >> 2) + sigma[c & 3] for c in h[order]])
        for pol in (bl.basic_policy, bl.strategy_policy):
            r1, r2 = bl.regret(h, pol(h)), bl.regret(h2, pol(h2))
            assert abs(r1 - r2) < 1e-12


def test_strategy_optimal_on_representative_hands():
    for row in bl.representative_table():
        _, _, sev = row["strategy"]
        _, _, oev = row["optimal"]
        assert abs(sev - oev) < 1e-12, row["hand"]


def test_oracle_matches_exact_argmax():
    pol = bl.oracle_policy()
    for text in ["Jh Qh Kh Ah Jd", "3h 5h 9h Kh 5d", "2h 7s 8d Tc Jh"]:
        h = pe.parse_hand(text)
        q = pe.exact_action_values(h)
        assert abs(q[pol(h)] - q.max()) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
