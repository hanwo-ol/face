
import json
import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
import argparse, os
from pathlib import Path

def _resolve_data_dir():
    # Priority: --data-dir arg > SIAMESE_DATA_DIR env > current directory
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=str, default=None)
    # Parse known args only to avoid interfering with existing parsers
    args, _ = parser.parse_known_args()
    cand = args.data_dir or os.environ.get("SIAMESE_DATA_DIR", None)
    return Path(cand) if cand else Path(".")
DATA_DIR = _resolve_data_dir()
IN_FILE = DATA_DIR / "preprocessed_texts.json"
OUT_FILE = DATA_DIR / "siamese_data_pairs.json"
SAME_TEXT_SELF_PAIR_ALLOWED = False  # <- improvement: don't allow (A, A) pairs by default
SYMMETRY_DEDUP = True  # <- improvement: avoid adding both (A,B) and (B,A)

def make_pairs(humans, ais, negative_ratio=1.0):
    """Create positive and negative pairs with balance.
    Positive = (human,human) + (ai,ai), Negative = (human,ai).
    Ensures no (A,A) unless explicitly allowed; optionally de-duplicate symmetric duplicates.
    """
    pos_pairs = []
    neg_pairs = []
    seen_pairs = set()

    # helper to add pair with optional symmetry dedup
    def _add_pair(a, b, label):
        if not SAME_TEXT_SELF_PAIR_ALLOWED and a == b and label == 1:
            return

        if SYMMETRY_DEDUP:
            key = tuple(sorted((a, b))) + (label,)
        else:
            key = (a, b, label)

        if key in seen_pairs:
            return
        seen_pairs.add(key)

        return {"a": a, "b": b, "label": label}

    # positive human-human
    for i in range(len(humans)):
        for j in range(i + 1, len(humans)):
            item = _add_pair(humans[i], humans[j], 1)
            if item: pos_pairs.append(item)

    # positive ai-ai
    for i in range(len(ais)):
        for j in range(i + 1, len(ais)):
            item = _add_pair(ais[i], ais[j], 1)
            if item: pos_pairs.append(item)

    # negative human-ai
    for h in humans:
        for a in ais:
            item = _add_pair(h, a, 0)
            if item: neg_pairs.append(item)

    # balance
    if len(neg_pairs) == 0 or len(pos_pairs) == 0:
        raise ValueError("Not enough pairs to balance. Check input sizes.")

    min_count = min(len(neg_pairs), len(pos_pairs))
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    pos_pairs = pos_pairs[:min_count]
    neg_pairs = neg_pairs[:min_count]

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    return pairs

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"{IN_FILE} not found. Run 01_load_and_prepare_data.py first.")
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    humans = data.get("human_texts", [])
    ais = data.get("ai_texts", [])

    pairs = make_pairs(humans, ais)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pairs)} pairs to {OUT_FILE}")

if __name__ == "__main__":
    main()
