"""Task 4 experiments: state abstraction and reward design.

    python task4_experiments.py --episodes 50000000 --workers 16   # the report's run (~25 min)
    python task4_experiments.py --quick                            # smoke test (about 1 min)
    python task4_experiments.py --plot-only                        # redraw figures from results/

Outputs
    results/task4/exact_analysis.json      exact ceilings and paytable comparisons
    results/task4/runs/<run>.json          one file per training run (CONVENTIONS.md format, plus a curve)
    report/figures/task4_*.png             figures used in report/task4_abstraction_rewards.md

Every policy is scored exactly with exact_values (no Monte Carlo noise): the
value of a hand is Q*(state, mask) under the main 9/6 table. ``eval_mean`` is
the average over the 50,000 held-out eval hands and ``deal_return`` the average
over a uniform deal.

The learner here is a small batched tabular Q-learner with the sample-average
step size alpha = 1/N(s, a) and linearly decayed epsilon-greedy exploration.
Episodes end after the draw, so the update is a running mean of rewards and
processing a batch at once gives the same table as one-at-a-time updates; only
the greedy choices inside a batch use the table from the start of the batch.
It is only here to compare abstractions and rewards under one fixed setting;
the step-size and exploration sweeps belong to Task 2's agent.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import abstraction as A
import exact_values as ev
import paytables as pt
import poker_env as pe
from poker_data import load_split

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "task4")
FIGURES = os.path.join(ROOT, "report", "figures")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(canon_masks, eval_split) -> dict:
    """Exact scores of a per-state policy under the main paytable."""
    vals = ev.policy_values(canon_masks)
    per_hand = vals[eval_split["state_id"]]
    cls = eval_split["uci_class"]
    return {
        "eval_mean": float(per_hand.mean()),
        "eval_ci95": float(1.96 * per_hand.std(ddof=1) / np.sqrt(len(per_hand))),
        "deal_return": float(vals @ ev.state_weights()),
        "by_class": {pe.UCI_CLASS_NAMES[c]: float(per_hand[cls == c].mean()) for c in np.unique(cls)},
    }


def hold_str(state_id: int, canon_mask: int) -> str:
    cards = pe.canonical_states()[state_id]
    kept = [pe.card_to_str(int(cards[i])) for i in pe.kept_positions(int(canon_mask))]
    return " ".join(kept) if kept else "(draw five)"


# --------------------------------------------------------------------------
# Exact analysis
# --------------------------------------------------------------------------
def exact_analysis(eval_split) -> dict:
    opt_masks = ev.optimal_masks()
    opt = score(opt_masks, eval_split)
    w = ev.state_weights()
    q_main = ev.q_table()

    abstractions = {}
    for name in A.ABSTRACTIONS:
        ab = A.get_abstraction(name)
        s = score(ab.best_canonical_masks(), eval_split)
        abstractions[name] = {
            "n_states": ab.n_states, "n_actions": ab.n_actions,
            "table_entries": ab.n_states * ab.n_actions,
            "ceiling_deal_return": s["deal_return"], "ceiling_eval_mean": s["eval_mean"],
            "pct_of_optimal": 100 * s["deal_return"] / opt["deal_return"],
        }

    paytables = {}
    for name, table in pt.PAYTABLES.items():
        masks = ev.optimal_masks(table)
        vals = ev.policy_values(masks)
        second = ev.second_moment_table()[np.arange(len(masks)), masks] @ w
        mean = vals @ w
        differs = masks != opt_masks
        loss = w * (q_main[np.arange(len(masks)), opt_masks] - vals)
        examples = [{
            "hand": pe.hand_to_str(pe.canonical_states()[s]),
            "hold_9_6": hold_str(s, opt_masks[s]), f"hold_{name}": hold_str(s, masks[s]),
            "value_lost_9_6": float(q_main[s, opt_masks[s]] - vals[s]),
        } for s in np.argsort(-loss)[:3] if differs[s]]
        win_vals = ev.policy_values(masks, pt.WIN_ONLY)
        # Some rewards make several holds equally good; how ties are broken changes the 9/6 score.
        q_own = ev.q_table(table)
        tied = q_own >= q_own.max(axis=1, keepdims=True) - 1e-12
        tie_range = [float(np.where(tied, q_main, np.inf).min(axis=1) @ w),
                     float(np.where(tied, q_main, -np.inf).max(axis=1) @ w)]
        paytables[name] = {
            "kind": "game" if name in pt.GAMES else "training_reward",
            "payouts": {k: float(v) for k, v in table.items()},
            "optimal_return_own_table": ev.optimal_return(table),
            "return_on_9_6": float(mean),
            "return_on_9_6_over_ties": tie_range,
            "loss_vs_9_6_optimal": float(opt["deal_return"] - mean),
            "std_on_9_6": float(np.sqrt(second - mean ** 2)),
            "p_paying_hand": float(win_vals @ w),
            "pct_hands_hold_differs": float(100 * w[differs].sum()),
            "examples": examples,
        }
    return {"optimal_9_6": opt, "abstractions": abstractions, "paytables": paytables}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def epsilon(frac: float, start=1.0, end=0.05, decay_frac=0.5) -> float:
    return end + (start - end) * max(0.0, 1 - frac / decay_frac)


def train(cfg: dict) -> dict:
    """One training run. cfg: abstraction, reward, episodes, seed, batch, checkpoints."""
    t0 = time.time()
    ab = A.get_abstraction(cfg["abstraction"])
    reward = pe.paytable_vector(pt.get_paytable(cfg["reward"]))
    train_states = load_split("train")["state_id"].astype(np.int64)
    eval_split = load_split("eval")
    canon = pe.canonical_states().astype(np.int64)
    rng = np.random.default_rng(cfg["seed"])

    ns, na = ab.n_states, ab.n_actions
    n_visits = np.zeros(ns * na, dtype=np.int64)
    r_sum = np.zeros(ns * na)
    q = np.zeros((ns, na))
    curve, done = [], 0
    checkpoints = sorted(cfg["checkpoints"])
    while done < cfg["episodes"]:
        b = min(cfg["batch"], cfg["episodes"] - done, checkpoints[0] - done if checkpoints else cfg["batch"])
        sid = train_states[rng.integers(0, len(train_states), b)]
        z = ab.state_map[sid].astype(np.int64)
        greedy = np.argmax(q[z] + 1e-9 * rng.random((b, na)), axis=1)  # random tie-break
        explore = rng.random(b) < epsilon(done / cfg["episodes"])
        a = np.where(explore, rng.integers(0, na, b), greedy)
        final = pe.simulate_draw_batch(canon[sid], ab.action_map[sid, a], rng)
        r = reward[pe.hand_category_batch(final)]
        flat = z * na + a
        n_visits += np.bincount(flat, minlength=ns * na)
        r_sum += np.bincount(flat, weights=r, minlength=ns * na)
        q = (r_sum / np.maximum(n_visits, 1)).reshape(ns, na)
        done += b
        if checkpoints and done == checkpoints[0]:
            checkpoints.pop(0)
            s = score(ab.greedy_canonical_masks(q), eval_split)
            curve.append({"episodes": done, "eval_mean": s["eval_mean"], "deal_return": s["deal_return"],
                          "pct_pairs_visited": float(100 * (n_visits > 0).mean())})
    final_score = score(ab.greedy_canonical_masks(q), eval_split)
    return {"config": cfg, "seed": cfg["seed"], "episodes": done, **final_score,
            "runtime_s": time.time() - t0, "curve": curve}


def run_name(cfg) -> str:
    return f"{cfg['abstraction']}__{cfg['reward']}__seed{cfg['seed']}"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot(runs, exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES, exist_ok=True)
    opt = exact["optimal_9_6"]["eval_mean"]
    colors = {"canonical": "C0", "ranks": "C1", "coarse": "C2",
              "job_9_6": "C0", "job_9_6_clip25": "C3", "job_9_6_log1p": "C4", "win_only": "C5"}

    def curves(abstraction, reward):
        group = [r for r in runs if r["config"]["abstraction"] == abstraction and r["config"]["reward"] == reward]
        if not group:
            return None
        x = np.array([p["episodes"] for p in group[0]["curve"]])
        y = np.array([[p["eval_mean"] for p in r["curve"]] for r in group])
        return x, y.mean(0), y.std(0), len(group)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name in A.ABSTRACTIONS:
        c = curves(name, "job_9_6")
        if c is None:
            continue
        x, m, s, k = c
        info = exact["abstractions"][name]
        ax.plot(x, m, color=colors[name], label=f"{name} ({info['n_states']:,} x {info['n_actions']})")
        ax.fill_between(x, m - s, m + s, color=colors[name], alpha=0.2)
        ax.axhline(info["ceiling_eval_mean"], color=colors[name], ls=":", lw=1)
    ax.axhline(opt, color="k", ls="--", lw=1, label=f"optimal ({opt:.4f})")
    ax.plot([], [], color="0.4", ls=":", lw=1, label="best policy the abstraction can express")
    ax.set(xscale="log", xlabel="training episodes", ylabel="expected payout on eval hands (9/6)",
           title="State abstraction: learning speed vs. ceiling", ylim=(0.3, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "task4_abstraction_curves.png"), dpi=150)
    plt.close(fig)

    # Reward transforms on the coarse abstraction (the canonical table is still far from
    # converged at this budget, so its reward curves overlap; see the run JSONs).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ylim, xmin in zip(axes, ((0.6, 1.0), (0.93, 0.995)), (None, 1e5)):
        for reward in ("job_9_6", "job_9_6_clip25", "job_9_6_log1p", "win_only"):
            c = curves("coarse", reward)
            if c is None:
                continue
            x, m, s, k = c
            ax.plot(x, m, color=colors[reward], label=reward, marker=".", ms=4)
            ax.fill_between(x, m - s, m + s, color=colors[reward], alpha=0.2)
        ax.axhline(opt, color="k", ls="--", lw=1, label="optimal")
        ax.axhline(exact["abstractions"]["coarse"]["ceiling_eval_mean"], color="C2", ls=":", lw=1,
                   label="coarse ceiling")
        ax.set(xscale="log", xlabel="training episodes", ylim=ylim)
        if xmin:
            ax.set_xlim(left=xmin)
    axes[0].set(ylabel="expected payout on eval hands (9/6)", title="coarse abstraction, all rewards")
    axes[1].set(title="zoom: 9/6 vs. clipped vs. log rewards")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Training reward vs. score on the 9/6 table")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "task4_reward_curves.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    ap.add_argument("--episodes", type=int, default=20_000_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved results")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds = 200_000, 1

    if args.plot_only:
        runs = []
        for fname in sorted(os.listdir(os.path.join(RESULTS, "runs"))):
            with open(os.path.join(RESULTS, "runs", fname)) as f:
                runs.append(json.load(f))
        with open(os.path.join(RESULTS, "exact_analysis.json")) as f:
            plot(runs, json.load(f))
        return

    os.makedirs(os.path.join(RESULTS, "runs"), exist_ok=True)
    A.get_abstraction("coarse")  # build caches once before forking workers
    eval_split = load_split("eval")

    print("exact analysis ...")
    exact = exact_analysis(eval_split)
    with open(os.path.join(RESULTS, "exact_analysis.json"), "w") as f:
        json.dump(exact, f, indent=2)

    checkpoints = sorted({int(x) for x in np.geomspace(1e4, args.episodes, 15).round(-3)} | {args.episodes})
    base = dict(episodes=args.episodes, batch=50_000, checkpoints=checkpoints)
    configs = []
    for seed in range(args.seeds):
        for name in A.ABSTRACTIONS:
            configs.append(dict(base, abstraction=name, reward="job_9_6", seed=seed))
        for name in ("canonical", "coarse"):
            for reward in ("job_9_6_clip25", "job_9_6_log1p", "win_only"):
                configs.append(dict(base, abstraction=name, reward=reward, seed=seed))

    print(f"{len(configs)} training runs on {args.workers} workers ...")
    runs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(train, configs):
            runs.append(res)
            with open(os.path.join(RESULTS, "runs", run_name(res["config"]) + ".json"), "w") as f:
                json.dump(res, f, indent=2)
            print(f"  {run_name(res['config']):40s} eval_mean={res['eval_mean']:.4f} ({res['runtime_s']:.0f}s)")

    plot(runs, exact)
    print("done")


if __name__ == "__main__":
    main()
