
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import ops
from keras.utils import register_keras_serializable
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
DATA_DIR = _resolve_data_dir()# Paths
TOKENIZED_FILE = DATA_DIR / "tokenized_data.npz"
CONFIG_FILE = DATA_DIR / "model_config.json"
SIAMESE_MODEL_FILE = DATA_DIR / "siamese_model.keras"
SPLIT_FILE = DATA_DIR / "train_val_test_split_data.npz"

# Hyperparams for loss
MARGIN = 1.0

@register_keras_serializable(package="custom")
def euclidean_distance(vects):
    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, ops.epsilon()))

@register_keras_serializable(package="custom")
def manhattan_distance(vects):
    x, y = vects
    return ops.sum(ops.abs(x - y), axis=1, keepdims=True)

@register_keras_serializable(package="custom")
def cosine_distance(vects):
    x, y = vects
    x = ops.nn.l2_normalize(x, axis=1)
    y = ops.nn.l2_normalize(y, axis=1)
    # cosine similarity in [-1,1], distance = 1 - cos_sim in [0,2]
    cos_sim = ops.sum(x * y, axis=1, keepdims=True)
    return 1.0 - cos_sim

@register_keras_serializable(package="custom")
def contrastive_loss(y_true, y_pred):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    pos_loss = y_true * ops.square(y_pred)
    neg_loss = (1.0 - y_true) * ops.square(ops.maximum(0.0, MARGIN - y_pred))
    return ops.mean(pos_loss + neg_loss)

def main():
    if not TOKENIZED_FILE.exists():
        raise FileNotFoundError(f"{TOKENIZED_FILE} not found. Run 03_tokenize_data.py first.")
    if not SIAMESE_MODEL_FILE.exists():
        raise FileNotFoundError(f"{SIAMESE_MODEL_FILE} not found. Run 05_siamese_model.py first.")

    data = np.load(TOKENIZED_FILE)
    X_pairs = data["X_pairs"]  # shape (N, 2, max_len)
    y = data["y"]
    X_a = X_pairs[:, 0, :]
    X_b = X_pairs[:, 1, :]

    # 3-way split: train / val / test = 64% / 16% / 20%
    X_a_train, X_a_temp, X_b_train, X_b_temp, y_train, y_temp = train_test_split(
        X_a, X_b, y, test_size=0.36, random_state=42, stratify=y
    )
    X_a_val, X_a_test, X_b_val, X_b_test, y_val, y_test = train_test_split(
        X_a_temp, X_b_temp, y_temp, test_size=20/36, random_state=42, stratify=y_temp
    )

    np.savez_compressed(
        SPLIT_FILE,
        X_a_train=X_a_train, X_b_train=X_b_train, y_train=y_train,
        X_a_val=X_a_val, X_b_val=X_b_val, y_val=y_val,
        X_a_test=X_a_test, X_b_test=X_b_test, y_test=y_test
    )
    print(f"Saved 3-way split to {SPLIT_FILE}")

    # compile (load and re-save compiled model)
    model = keras.models.load_model(
        SIAMESE_MODEL_FILE,
        custom_objects={
            "euclidean_distance": euclidean_distance,
            "manhattan_distance": manhattan_distance,
            "cosine_distance": cosine_distance,
            "contrastive_loss": contrastive_loss,
        },
        compile=False,
    )
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# --- Optional helper override ---
helper = DATA_DIR / "06_compile.json"
if helper.exists():
    try:
        hp = json.loads(helper.read_text())
        margin = float(hp.get("margin", MARGIN))
        lr = float(hp.get("learning_rate", 1e-3))
        globals()["MARGIN"] = margin
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    except Exception as e:
        print("Warning: failed to load 06_compile.json:", e)

    model.compile(optimizer=optimizer, loss=contrastive_loss)

    # Save the compiled model back
    model.save(SIAMESE_MODEL_FILE)
    print(f"Compiled and saved model to {SIAMESE_MODEL_FILE}")

if __name__ == "__main__":
    main()
