"""Exact check of the baseline policies against the optimal policy (Task 3).

For a random sample of eval hands, computes every policy's EXACT expected payout
(no simulation noise) from pe.exact_action_values, the regret versus the oracle,
and which strategy rules lose EV. Final Monte Carlo evaluation of all policies is
Task 5's runner; this script only checks the decision rules.

    python baselines_check.py [n_hands]      # default 3000, ~5 min the first time
"""

import sys
import time
from collections import defaultdict

import numpy as np

import baselines as bl
import poker_env as pe
from poker_data import load_split

SAMPLE_SEED = 3


def main(n=3000):
    ev = load_split("eval")
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(ev["cards"]), n, replace=False)
    hands = ev["cards"][idx]
    classes = ev["uci_class"][idx]

    det = {"keep all": bl.keep_all, "pairs": bl.pairs_policy,
           "basic": bl.basic_policy, "strategy": bl.strategy_policy}
    vals = defaultdict(list)
    rule_stats = defaultdict(lambda: [0, 0, 0.0])  # fired, suboptimal, total regret
    worst = []
    t0 = time.time()
    print(f"Checking {n} eval hands against the exact optimum...", flush=True)
    for k, (h, c) in enumerate(zip(hands, classes), 1):
        canon, perm, sid = pe.canonicalize(h)
        best, q = bl.canonical_best(sid, np.array(canon))
        vals["oracle"].append(q[best])
        vals["random"].append(q.mean())  # uniform random mask, exact
        for name, pol in det.items():
            vals[name].append(q[pe.original_mask_to_canonical(pol(h), perm)])
        mask, rule = bl.explain(h, bl.STRATEGY_RULES)
        loss = q[best] - q[pe.original_mask_to_canonical(mask, perm)]
        st = rule_stats[rule]
        st[0] += 1
        if loss > 1e-9:
            st[1] += 1
            st[2] += loss
            worst.append((loss, pe.hand_to_str(h), rule, pe.hand_to_str(h[pe.kept_positions(mask)]),
                          pe.hand_to_str(h[pe.kept_positions(pe.canonical_mask_to_original(best, perm))])))
        if k % 100 == 0 or k == n:
            el = time.time() - t0
            print(f"  {k:>5}/{n}  {el:4.0f}s elapsed, ~{el / k * (n - k):3.0f}s left | running mean payout: "
                  f"basic {np.mean(vals['basic']):.3f}, strategy {np.mean(vals['strategy']):.3f}, "
                  f"oracle {np.mean(vals['oracle']):.3f}", flush=True)
    print()
    runtime = time.time() - t0

    oracle = np.array(vals["oracle"])
    print(f"{n} eval hands (sample seed {SAMPLE_SEED}), {runtime:.0f} s\n")
    print("| Policy | Exact mean payout | ± 95% CI (hand sampling) | Mean regret | Optimal on |")
    print("|---|---:|---:|---:|---:|")
    for name in ["random", "keep all", "pairs", "basic", "strategy", "oracle"]:
        v = np.array(vals[name])
        ci = 1.96 * v.std(ddof=1) / np.sqrt(n)
        r = oracle - v
        print(f"| {name} | {v.mean():.4f} | {ci:.4f} | {r.mean():.4f} | {np.mean(r < 1e-9):.1%} |")

    print("\nStrategy rules on this sample:\n")
    print("| Rule | Hands | Suboptimal | Total regret |")
    print("|---|---:|---:|---:|")
    order = [name for name, _ in bl.STRATEGY_RULES] + ["discard all"]
    for rule in order:
        f, s, tot = rule_stats.get(rule, [0, 0, 0.0])
        if f:
            print(f"| {rule} | {f} | {s} | {tot:.3f} |")

    print("\nLargest strategy mistakes:\n")
    print("| Hand | Rule | Strategy holds | Optimal holds | EV lost |")
    print("|---|---|---|---|---:|")
    for loss, h, rule, held, best in sorted(worst, reverse=True)[:12]:
        print(f"| {h} | {rule} | {held or '(none)'} | {best or '(none)'} | {loss:.3f} |")

    print("\nBy UCI class (exact mean payout):\n")
    names = ["random", "keep all", "pairs", "basic", "strategy", "oracle"]
    print("| Class | n | " + " | ".join(names) + " |")
    print("|---|---:|" + "---:|" * len(names))
    for c in np.unique(classes):
        m = classes == c
        cells = " | ".join(f"{np.array(vals[k])[m].mean():.3f}" for k in names)
        print(f"| {pe.UCI_CLASS_NAMES[c]} | {m.sum()} | {cells} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
