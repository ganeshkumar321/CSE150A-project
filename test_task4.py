"""Tests for Task 4 (paytables, exact values, abstractions).

Run with `python test_task4.py` (or pytest). Takes about a minute the first time.
"""

import numpy as np

import abstraction as A
import exact_values as ev
import paytables as pt
import poker_env as pe


def test_paytables_valid():
    for name, table in pt.PAYTABLES.items():
        pt.check_paytable(table)
    assert pt.get_paytable("job_9_6") is pe.JACKS_OR_BETTER


def test_exact_matches_enumeration():
    rng = np.random.default_rng(0)
    q = ev.q_table()
    for i in rng.integers(0, pe.N_HANDS, 4):
        canon, _, sid = pe.canonicalize(pe.all_hands()[i].astype(np.int64))
        assert np.allclose(pe.exact_action_values(np.array(canon)), q[sid], atol=1e-9)
    # a non-default paytable goes through the same path
    h = pe.parse_hand("Th Jh Qh Kh 2c")
    canon, _, sid = pe.canonicalize(h)
    assert np.allclose(pe.exact_action_values(np.array(canon), pt.JOB_8_5), ev.q_table(pt.JOB_8_5)[sid])


def test_published_returns():
    # Optimal-play returns for full-pay (9/6) and 8/5 Jacks or Better.
    assert abs(ev.optimal_return(pt.JOB_9_6) - 0.995439) < 1e-6
    assert abs(ev.optimal_return(pt.JOB_8_5) - 0.972984) < 1e-6
    assert abs(ev.state_weights().sum() - 1) < 1e-12


def test_known_holds():
    q = ev.q_table()
    _, _, sid = pe.canonicalize(pe.parse_hand("Jc Jd 2h 5s 8c"))
    assert abs(q[sid].max() - 1.537) < 1e-3          # keep JJ (value from Task 1)
    _, _, sid = pe.canonicalize(pe.parse_hand("Jh Qh Kh Ah 2c"))
    assert abs(q[sid].max() - 18.55) < 1e-2          # draw to the 4-card royal (value from Task 1)


def test_abstraction_shapes():
    sizes = {"canonical": (134459, 32), "ranks": (6175, 32), "coarse": (484, 11)}
    for name, (ns, na) in sizes.items():
        ab = A.get_abstraction(name)
        assert (ab.n_states, ab.n_actions) == (ns, na), name
        assert ab.state_map.shape == (pe.N_CANONICAL_STATES,)
        assert ab.action_map.min() >= 0 and ab.action_map.max() <= 31


def test_batch_matches_scalar():
    rng = np.random.default_rng(1)
    hands = np.argsort(rng.random((300, 52)), axis=1)[:, :5]
    for name in A.ABSTRACTIONS:
        ab = A.get_abstraction(name)
        actions = rng.integers(0, ab.n_actions, len(hands))
        z, sid, perm = ab.encode_batch(hands)
        masks = ab.to_mask_batch(actions, sid, perm)
        for h, a, zb, mb in zip(hands, actions, z, masks):
            zs, s, p = ab.encode(h)
            assert zs == zb and ab.to_mask(a, s, p) == mb


def test_masks_keep_the_right_cards():
    ab = A.get_abstraction("coarse")
    cases = {  # hand, rule, cards that must be kept
        ("8c Jd 3h Js 8h", "pairs"): {"8c", "Jd", "Js", "8h"},
        ("2h 9h Kc 5h Th", "flush_draw"): {"2h", "9h", "5h", "Th"},
        ("Kd 2c Qs Jh Td", "royal_draw"): {"Kd", "Td"},
        ("5c 6d 7h 8s Kc", "straight_draw"): {"5c", "6d", "7h", "8s"},
        ("Ac 7d Kh 2s Qc", "high_top2"): {"Ac", "Kh"},
    }
    for (s, rule), want in cases.items():
        hand = pe.parse_hand(s)
        z, sid, perm = ab.encode(hand)
        mask = ab.to_mask(A.HOLD_RULES.index(rule), sid, perm)
        kept = {pe.card_to_str(int(hand[i])) for i in pe.kept_positions(mask)}
        assert kept == want, (s, rule, kept)


def test_mapped_masks_have_canonical_value():
    # Playing an abstract action on a raw hand must be worth what the canonical table says.
    rng = np.random.default_rng(2)
    ab = A.get_abstraction("coarse")
    q = ev.q_table()
    for _ in range(3):
        hand = np.sort(rng.choice(52, 5, replace=False))[rng.permutation(5)]
        a = int(rng.integers(ab.n_actions))
        z, sid, perm = ab.encode(hand)
        assert abs(pe.exact_action_value(hand, ab.to_mask(a, sid, perm)) - q[sid, ab.action_map[sid, a]]) < 1e-9


def test_encoder_works_with_task2_agent():
    from q_learning import QLearningAgent, train

    ab = A.get_abstraction("ranks")
    agent = QLearningAgent(encoder=ab.state_encoder(), n_states=ab.n_states, state_encoding=ab.name, seed=0)
    hands = np.argsort(np.random.default_rng(3).random((50, 52)), axis=1)[:, :5]
    train(agent, hands, 500, paytable=pt.JOB_9_6_CLIP25)
    assert agent.Q.shape == (ab.n_states, 32) and agent.N.sum() == 500
    assert 0 <= agent(hands[0]) <= 31
    try:
        A.get_abstraction("coarse").state_encoder()
        assert False, "coarse uses hold rules, not 32 masks"
    except ValueError:
        pass


def test_ceilings_ordered():
    opt = ev.optimal_return()
    ceil = {n: ev.expected_return(A.get_abstraction(n).best_canonical_masks()) for n in A.ABSTRACTIONS}
    assert abs(ceil["canonical"] - opt) < 1e-12
    assert ceil["ranks"] < ceil["coarse"] < opt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
