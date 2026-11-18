"""UCI Poker Hand loading, deduplication and the shared train/dev/eval split (Task 1).

Split definition (all splits are disjoint as unordered 5-card sets):
    dev   : unique card sets of poker-hand-training-true.data (25,010 rows, biased:
            royal/straight flushes oversampled) -> development and tuning only.
    eval  : EVAL_SIZE card sets sampled (stratified by CLASS, seed SPLIT_SEED) from the
            unique card sets of poker-hand-testing.data that are not in dev.
    train : every remaining unique card set of the test file.
Each split keeps the first occurrence of a card set with its original card order.

Usage by other tasks:
    from poker_data import load_split
    s = load_split("train")   # dict: cards (N,5) int8, uci_class, state_id, source_row
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import poker_env as pe

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
TRAIN_FILE = "poker-hand-training-true.data"
TEST_FILE = "poker-hand-testing.data"
COLUMNS = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS"]

SPLIT_SEED = 150
EVAL_SIZE = 50_000
SPLITS = ("train", "dev", "eval")


def load_uci(name: str, raw_dir: str = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(raw_dir, name), header=None, names=COLUMNS)


def df_to_cards(df: pd.DataFrame) -> np.ndarray:
    """(N,5) card ints in the record's original card order."""
    suits = df[["S1", "S2", "S3", "S4", "S5"]].to_numpy()
    ranks = df[["C1", "C2", "C3", "C4", "C5"]].to_numpy()
    return pe.uci_to_cards(suits, ranks)


def first_unique(keys: np.ndarray) -> np.ndarray:
    """Row indices of the first occurrence of each key, in file order."""
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def stratified_sample(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `size` indices without replacement, proportional to label frequency
    (largest-remainder rounding)."""
    classes, counts = np.unique(labels, return_counts=True)
    quota = counts * size / counts.sum()
    take = np.floor(quota).astype(int)
    rest = size - take.sum()
    take[np.argsort(-(quota - take), kind="stable")[:rest]] += 1
    picked = [rng.choice(np.flatnonzero(labels == c), t, replace=False)
              for c, t in zip(classes, take) if t > 0]
    return np.sort(np.concatenate(picked))


def build_splits(seed: int = SPLIT_SEED, eval_size: int = EVAL_SIZE, out_dir: str = SPLIT_DIR):
    rng = np.random.default_rng(seed)
    dfs = {"uci_train": load_uci(TRAIN_FILE), "uci_test": load_uci(TEST_FILE)}
    cards = {k: df_to_cards(v) for k, v in dfs.items()}
    keys = {k: pe.hand_index_batch(v) for k, v in cards.items()}

    dev_rows = first_unique(keys["uci_train"])
    dev_keys = set(keys["uci_train"][dev_rows].tolist())

    test_rows = first_unique(keys["uci_test"])
    in_dev = np.isin(keys["uci_test"][test_rows], list(dev_keys))
    pool = test_rows[~in_dev]
    pool_class = dfs["uci_test"]["CLASS"].to_numpy()[pool]
    eval_rows = pool[stratified_sample(pool_class, eval_size, rng)]
    train_rows = np.setdiff1d(pool, eval_rows)

    plan = {"train": ("uci_test", train_rows), "dev": ("uci_train", dev_rows),
            "eval": ("uci_test", eval_rows)}
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"seed": seed, "eval_size": eval_size,
                "card_encoding": "card = 4*rank + suit; rank 0..12 = 2..A; suit 0..3 = h,s,d,c",
                "dedup_key": "unordered 5-card set (poker_env.hand_index)",
                "splits": {}}
    for name, (src, rows) in plan.items():
        c = cards[src][rows]
        data = dict(cards=c.astype(np.int8),
                    uci_class=dfs[src]["CLASS"].to_numpy()[rows].astype(np.int8),
                    state_id=pe.state_id_batch(c).astype(np.int32),
                    source_row=rows.astype(np.int32))
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **data)
        manifest["splits"][name] = {
            "source_file": TRAIN_FILE if src == "uci_train" else TEST_FILE,
            "n_hands": int(len(rows)),
            "n_states": int(len(np.unique(data["state_id"]))),
            "class_counts": {int(k): int(v) for k, v in
                             zip(*np.unique(data["uci_class"], return_counts=True))},
        }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split(name: str, split_dir: str = SPLIT_DIR) -> dict:
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    with np.load(os.path.join(split_dir, f"{name}.npz")) as z:
        out = {k: z[k] for k in z.files}
    out["cards"] = out["cards"].astype(np.int64)
    return out


if __name__ == "__main__":
    print(json.dumps(build_splits(), indent=2))
