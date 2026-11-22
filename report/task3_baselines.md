### Baseline strategies

**Purpose.** Baselines bracket what the Q-learning agent should achieve. Two reference policies set the floor, hand-written heuristics show how much simple poker knowledge gets you, and an exact oracle gives the ceiling. All are implemented in `baselines.py` behind the shared interface `policy(hand) -> mask` and need no training. The heuristics read only ranks and suits, so they are deterministic and treat every hand in a canonical state the same way.

**Reference policies.**
- *Random* picks one of the 32 masks uniformly, using its own seeded generator.
- *Keep all* stands pat, so its payout equals the payout of the dealt hand.

**Heuristic policies.** Each heuristic is an ordered list of rules. The first rule that matches decides which cards to hold, and if no rule matches the policy discards all five. Priorities matter whenever a hand fits more than one rule, e.g. a pair inside a four-card flush.

- *Pairs* keeps any hand that already scores: made straights or better are kept whole, otherwise the cards of the quads, trips, two pair or pair are kept. All other hands discard everything.
- *Basic* is the rule set the proposal describes: made hand > 4 to a flush > 4 to an open-ended straight > 4 to an inside straight > discard all. Any pair, even a low one, outranks every draw.
- *Strategy* is a 16-rule list ordered by the exact expected payout of each holding under 9/6 Jacks-or-Better (values from `pe.exact_action_values`):

| # | Hold | Typical EV | Why it sits here |
|---:|---|---:|---|
| 1 | Royal / straight flush (all 5) | 800 / 50 | Cannot be improved. |
| 2 | Four of a kind (the 4) | 25 | |
| 3 | Full house (all 5) | 9 | |
| 4 | 4 to a royal flush | ≈18.5 | Worth breaking a made flush (6), straight (4) or high pair (1.54). |
| 5 | Flush (all 5) | 6 | |
| 6 | Three of a kind (the 3) | 4.30 | |
| 7 | Straight (all 5) | 4 | |
| 8 | 4 to a straight flush | ≈2.1–3.6 | Beats a high pair; e.g. 8h 9h Th Jh is worth 3.57 vs 1.54 for J J. |
| 9 | Two pair (the 4) | 2.60 | |
| 10 | High pair (J or better) | 1.54 | Beats a 4-card flush (≈1.2). |
| 11 | 3 to a royal flush | ≈1.3–1.4 | Beats a low pair and a 4-card flush. |
| 12 | 4 to a flush | ≈1.2 | Beats a low pair (0.82). The *basic* policy gets this wrong. |
| 13 | Low pair (2s–Ts) | 0.82 | Beats an open-ended straight draw (≈0.68). |
| 14 | 4 to an open-ended straight | ≈0.68–0.87 | 2345 up to TJQK. A234 and JQKA complete at only one end, so they count as inside draws. |
| 15 | High cards (J+), at most two | ≈0.45–0.61 | Prefer two suited high cards, otherwise keep the two lowest ranks to leave room for a straight. |
| 16 | Discard all | ≈0.36 | An inside straight draw without high cards (0.34) is worse than drawing five new cards. |

Several pairs of rules can never apply to the same hand (a flush cannot contain three of a kind, for example). Those are ordered by EV only so the list reads top to bottom.

**Oracle.** `oracle_policy(paytable)` enumerates every possible draw for all 32 masks and holds the mask with the highest exact expected payout. It caches results per canonical state, so each new state costs about 0.1 s. It is the true optimum for any paytable, which makes it the upper bound for every learned agent and a ground truth for measuring how much each heuristic loses.

**Checking decisions on representative hands.** We built 23 hands that cover each rule and each priority conflict and compared every policy's hold with the exact optimum (`python baselines.py` prints the full table; `test_baselines.py` asserts each case). The strategy policy holds the optimal cards on all 23. The conflict cases show where simpler rules fail:

| Hand | Basic holds (EV) | Strategy holds (EV) = optimal |
|---|---|---|
| Th Jh Qh Kh 3h | whole flush (6.00) | Th Jh Qh Kh (19.55) |
| Jh Qh Kh Ah Jd | Jh Jd (1.54) | Jh Qh Kh Ah (18.53) |
| 8h 9h Th Jh Jd | Jh Jd (1.54) | 8h 9h Th Jh (3.57) |
| Th Qh Ah 4c 4d | 4c 4d (0.82) | Th Qh Ah (1.34) |
| 3h 5h 9h Kh 5d | 5h 5d (0.82) | 3h 5h 9h Kh (1.21) |
| 6h 6s 7d 8c 9h | 6h 6s (0.82) | 6h 6s (0.82) |
| 2h 7s 8d Tc Jh | 7s 8d Tc Jh (0.40) | Jh (0.48) |
| 2h 3s 5d 6c 9h | 2h 3s 5d 6c (0.34) | discard all (0.36) |
| Jh Qh As 4c 7d | discard all (0.31) | Jh Qh (0.61) |

**Exact comparison with the optimum.** On 3,000 randomly sampled eval hands we computed each policy's *exact* expected payout from the enumerated action values, so the numbers carry no draw noise. The CI reflects only which hands were sampled, and the policies are compared on the same hands. Final Monte Carlo results over the full eval split come from the Task 5 runner.

| Policy | Exact mean payout | ± 95% CI | Mean regret vs. oracle | Optimal hold on |
|---|---:|---:|---:|---:|
| Random | 0.340 | 0.006 | 0.664 | 0.0% |
| Keep all | 0.341 | 0.029 | 0.663 | 0.6% |
| Pairs | 0.880 | 0.032 | 0.124 | 52.8% |
| Basic | 0.929 | 0.034 | 0.075 | 57.1% |
| Strategy | 1.002 | 0.033 | 0.002 | 95.3% |
| Oracle | 1.004 | 0.033 | 0 | 100% |

The oracle's mean agrees with the known return of optimal 9/6 Jacks-or-Better play (≈0.995) within sampling error. The gap from *pairs* to *basic* comes from drawing to flushes and straights when no pair is held. The larger gap from *basic* to *strategy* comes from the priority fixes in the table above, especially playing 4-card flushes over low pairs and holding high cards rather than discarding them. Broken down by starting class, all heuristics are optimal on two pair or better. Nearly all of the remaining regret is on "nothing" hands (strategy 0.567 vs. oracle 0.572) and on one-pair hands for *pairs* and *basic*.

**Where the strategy still loses.** Of the 3,000 hands, 142 are suboptimal, and 141 of them fall under the "high cards" or "discard all" rules. The biggest misses are three-card straight-flush draws, e.g. 7s 8s 9s from 8s 7s 4d 9s 2h (EV 0.275 better than discarding all) and Td 9d Jd over a lone Jd. Professional strategy charts add rules for these draws and for suited high cards with a ten, but their value depends on gaps and high cards. Adding them would lengthen the list a lot for at most 0.002 in mean payout, so we stop at 16 rules.

**Use in the experiments.** The heuristics are tuned for the main paytable `pe.JACKS_OR_BETTER`; under the alternative paytables (Task 4) some priorities may no longer be right, which is itself a useful contrast with a learned policy. The oracle accepts any paytable and stays optimal. Random and keep-all establish the floor, strategy shows what expert knowledge achieves, and the oracle is the ceiling a tabular Q-learner should approach as its Q-values converge.
