"""Unit tests for poker_env. Run with `python test_poker_env.py` (or pytest)."""

import itertools

import numpy as np

import poker_env as pe


def test_encoding_roundtrip():
    for c in range(52):
        assert pe.str_to_card(pe.card_to_str(c)) == c
    s, r = pe.cards_to_uci(np.arange(52))
    assert (pe.uci_to_cards(s, r) == np.arange(52)).all()
    assert pe.uci_to_cards([1], [1])[0] == pe.str_to_card("Ah")
    assert pe.uci_to_cards([4], [2])[0] == pe.str_to_card("2c")


def test_category_counts():
    counts = np.bincount(pe._tables()["category_lut"], minlength=11)
    assert counts.tolist() == [1302540, 760320, 337920, 123552, 54912, 10200, 5108, 3744, 624, 36, 4]


def test_edge_cases():
    cases = {"Th Jh Qh Kh Ah": "royal_flush", "Ah 2h 3h 4h 5h": "straight_flush",
             "As 2d 3c 4h 5s": "straight", "Qs Kd Ac 2h 3s": "nothing",
             "Jc Jd 2h 5s 8c": "jacks_or_better", "Tc Td 2h 5s 8c": "low_pair"}
    for s, want in cases.items():
        assert pe.CATEGORY_NAMES[pe.hand_category(pe.parse_hand(s))] == want, s


def test_reference_agrees():
    rng = np.random.default_rng(0)
    hands = np.argsort(rng.random((20000, 52)), axis=1)[:, :5]
    ref = [pe._reference_category(h.tolist()) for h in hands]
    assert (pe.hand_category_batch(hands) == ref).all()


def test_canonical_states():
    assert len(pe.canonical_states()) == pe.N_CANONICAL_STATES


def test_mask_mapping_preserves_value():
    rng = np.random.default_rng(1)
    for _ in range(50):
        h = rng.choice(52, 5, replace=False)
        canon, perm, _ = pe.canonicalize(h)
        cm = int(rng.choice([m for m in range(32) if bin(m).count("1") >= 3]))
        a = pe.exact_action_value(h, pe.canonical_mask_to_original(cm, perm))
        b = pe.exact_action_value(np.array(canon), cm)
        assert abs(a - b) < 1e-12


def test_mask_roundtrip():
    for p in itertools.permutations(range(5)):
        for m in range(32):
            assert pe.original_mask_to_canonical(pe.canonical_mask_to_original(m, p), p) == m


def test_simulator_structure():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        h = rng.choice(52, 5, replace=False)
        m = int(rng.integers(32))
        f = pe.simulate_draw(h, m, rng)
        assert len(set(f.tolist())) == 5
        for i in range(5):
            if (m >> i) & 1:
                assert f[i] == h[i]
            else:
                assert f[i] not in h
    hands = np.argsort(rng.random((5000, 52)), axis=1)[:, :5]
    masks = rng.integers(0, 32, 5000)
    f = pe.simulate_draw_batch(hands, masks, rng)
    keep = ((masks[:, None] >> np.arange(5)) & 1).astype(bool)
    assert (f[keep] == hands[keep]).all()
    assert not ((f[:, :, None] == hands[:, None, :]).any(2) & ~keep).any()


def test_simulator_matches_exact():
    rng = np.random.default_rng(3)
    h = pe.parse_hand("Jc Jd 2h 5s 8c")
    exact = pe.exact_action_value(h, 0b00011)
    r = pe.payout_batch(pe.simulate_draw_batch(np.repeat(h[None], 200000, 0), 0b00011, rng))
    assert abs(r.mean() - exact) < 4 * r.std() / np.sqrt(len(r))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
