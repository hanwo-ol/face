
"""
Automated hyperparameter experiments for the Siamese text model.
This script runs multiple configurations end-to-end from the tokenized data onward,
trains, evaluates, and aggregates a CSV/JSON summary.

Assumptions:
- 01~05 steps have been (or can be) run with their defaults to generate prerequisites:
  - preprocessed_texts.json
  - siamese_data_pairs.json
  - tokenized_data.npz, tokenizer.json, model_config.json
- This script performs its own 3-way split (via 06) and uses the improved 07 for training/eval.
- Language is English only; no special handling for Korean.

Outputs:
- experiments_summary.csv (one row per run)
- per-run JSON summaries under runs/<run_id>/evaluation_summary.json
- best_overall.json with the top run by test F1
"""

import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------- Configuration ----------------
import argparse, os
from pathlib import Path

def _resolve_data_dir():
    import argparse, os
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=str, default=None)
    args, _ = parser.parse_known_args()
    cand = args.data_dir or os.environ.get("SIAMESE_DATA_DIR", None)
    return Path(cand) if cand else Path(".")
DATA_DIR = _resolve_data_dir()
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Grid of experiments (kept reasonable; adjust as needed)
DISTANCES = ["euclidean", "manhattan", "cosine"]
MARGINS = [0.5, 1.0]
LRS = [1e-3, 3e-4]
RNN_TYPES = ["lstm", "gru"]
EMBED_DIMS = [128]
FEATURE_DIMS = [32, 64]
EPOCHS = 10
BATCH_SIZE = 128

# ---------------- Utilities ----------------
def run_py(path):
    print(f"==> Running {path}")
    subprocess.run([sys.executable, str(path)], check=True)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def timestamp_id():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

# ---------------- Orchestration steps ----------------
def ensure_until_tokenized():
    # Run 01-03 if their outputs don't exist.
    if not (DATA_DIR / "preprocessed_texts.json").exists():
        run_py(DATA_DIR / "01_load_and_prepare_data.py")
    if not (DATA_DIR / "siamese_data_pairs.json").exists():
        run_py(DATA_DIR / "02_pair_data_generator.py")
    if not (DATA_DIR / "tokenized_data.npz").exists():
        run_py(DATA_DIR / "03_tokenize_data.py")

def build_base_network(rnn_type, embed_dim, feature_dim):
    # Patch model_config.json to control hyperparams for 04.
    cfg_path = DATA_DIR / "model_config.json"
    cfg = read_json(cfg_path)
    cfg["RNN_TYPE"] = rnn_type.upper()  # expected by 04
    cfg["EMBED_DIM"] = embed_dim
    cfg["FEATURE_DIM"] = feature_dim
    write_json(cfg_path, cfg)
    run_py(DATA_DIR / "04_make_base_network.py")

def build_siamese(distance):
    # Patch siamese config if 05 reads from config; else rely on arg/env via JSON helper
    # We'll write a tiny helper file 05_distance.json read by 05 if present (non-breaking).
    helper = DATA_DIR / "05_distance.json"
    write_json(helper, {"distance": distance})
    run_py(DATA_DIR / "05_siamese_model.py")

def compile_and_split(margin, lr):
    # Margin and lr are controlled inside 06 (we keep margin constant for loss;
    # learning rate is baked at compile time). We'll write a helper file.
    helper = DATA_DIR / "06_compile.json"
    write_json(helper, {"margin": margin, "learning_rate": lr})
    run_py(DATA_DIR / "06_compiler_splitter.py")

def train_and_eval(epochs, batch_size):
    # 07 has constants; to avoid modifying file repeatedly, we write a small helper file.
    helper = DATA_DIR / "07_train.json"
    write_json(helper, {"epochs": epochs, "batch_size": batch_size})
    run_py(DATA_DIR / "07_train_and_evaluate.py")

def collect_metrics():
    summ = read_json(DATA_DIR / "evaluation_summary.json")
    return summ

def save_run_artifacts(run_dir, tags):
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save important artifacts
    artifacts = [
        "trained_best_model.keras",
        "siamese_model.keras",
        "training_curve.png",
        "evaluation_summary.json",
        "model_config.json",
        "tokenizer.json",
        "train_val_test_split_data.npz",
    ]
    for a in artifacts:
        p = DATA_DIR / a
        if p.exists():
            shutil.copy2(p, run_dir / a)
    write_json(run_dir / "tags.json", tags)

def append_csv_row(csv_path, header, row):
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")

def main():
    ensure_until_tokenized()

    results = []
    csv_path = DATA_DIR / "experiments_summary.csv"
    header = [
        "run_id","distance","margin","lr","rnn","embed_dim","feature_dim","epochs","batch_size",
        "val_best_thr","val_f1","val_roc_auc","val_pr_auc","val_acc",
        "test_f1","test_roc_auc","test_pr_auc","test_acc"
    ]

    for distance, margin, lr, rnn, embed_dim, feature_dim in itertools.product(
        DISTANCES, MARGINS, LRS, RNN_TYPES, EMBED_DIMS, FEATURE_DIMS
    ):
        run_id = timestamp_id()
        print(f"\n==== RUN {run_id} ====")
        tags = {
            "run_id": run_id,
            "distance": distance,
            "margin": margin,
            "learning_rate": lr,
            "rnn": rnn,
            "embed_dim": embed_dim,
            "feature_dim": feature_dim,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }

        # Build & train
        build_base_network(rnn, embed_dim, feature_dim)
        build_siamese(distance)
        compile_and_split(margin, lr)
        train_and_eval(EPOCHS, BATCH_SIZE)

        # Collect & persist
        summ = collect_metrics()
        run_dir = RUNS_DIR / run_id
        save_run_artifacts(run_dir, tags)

        # Extract metrics
        row = [
            run_id, distance, margin, lr, rnn, embed_dim, feature_dim, EPOCHS, BATCH_SIZE,
            summ["val"]["best_threshold"],
            summ["val"]["best_val_f1"], summ["val"]["roc_auc"], summ["val"]["pr_auc"], summ["val"]["accuracy"],
            summ["test"]["f1"], summ["test"]["roc_auc"], summ["test"]["pr_auc"], summ["test"]["accuracy"]
        ]
        append_csv_row(csv_path, header, row)
        results.append({"tags": tags, "metrics": summ})

    # Find best by test F1
    best = max(results, key=lambda r: r["metrics"]["test"]["f1"])
    write_json(DATA_DIR / "best_overall.json", best)
    print("Best run:", best["tags"]["run_id"], "Test F1:", best["metrics"]["test"]["f1"])

if __name__ == "__main__":
    main()
