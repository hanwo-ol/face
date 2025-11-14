
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras import ops
from keras.utils import register_keras_serializable
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, roc_curve, accuracy_score
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
DATA_DIR = _resolve_data_dir()SPLIT_FILE = DATA_DIR / "train_val_test_split_data.npz"
MODEL_FILE = DATA_DIR / "siamese_model.keras"
HISTORY_PNG = DATA_DIR / "training_curve.png"
SUMMARY_JSON = DATA_DIR / "evaluation_summary.json"

EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 3
MARGIN = 1.0  # must match loss

@register_keras_serializable(package="custom")
def contrastive_loss(y_true, y_pred):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    pos_loss = y_true * ops.square(y_pred)
    neg_loss = (1.0 - y_true) * ops.square(ops.maximum(0.0, MARGIN - y_pred))
    return ops.mean(pos_loss + neg_loss)

def compute_threshold(y_true, distances):
    """Choose threshold that maximizes F1 on validation set."""
    # Convert distance to similarity score in [0,1] by min-max on val for PR/ROC (optional)
    # For threshold search, we scan unique distances
    best_thr, best_f1 = None, -1.0
    # To avoid too many thresholds, sample quantiles
    qs = np.linspace(0, 1, 201)
    candidates = np.unique(np.quantile(distances, qs))
    for thr in candidates:
        y_pred = (distances < thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


# --- Optional helper override for epochs/batch_size ---
helper = DATA_DIR / "07_train.json"
if helper.exists():
    try:
        hp = json.loads(helper.read_text())
        globals()["EPOCHS"] = int(hp.get("epochs", EPOCHS))
        globals()["BATCH_SIZE"] = int(hp.get("batch_size", BATCH_SIZE))
    except Exception as e:
        print("Warning: failed to load 07_train.json:", e)


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} not found. Run 06_compiler_splitter.py first.")

    data = np.load(SPLIT_FILE)
    X_a_train, X_b_train, y_train = data["X_a_train"], data["X_b_train"], data["y_train"]
    X_a_val, X_b_val, y_val = data["X_a_val"], data["X_b_val"], data["y_val"]
    X_a_test, X_b_test, y_test = data["X_a_test"], data["X_b_test"], data["y_test"]

    model = keras.models.load_model(MODEL_FILE, custom_objects={"contrastive_loss": contrastive_loss})
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("trained_best_model.keras", monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        [X_a_train, X_b_train], y_train,
        validation_data=([X_a_val, X_b_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate using tuned threshold on val
    best_model = keras.models.load_model("trained_best_model.keras", custom_objects={"contrastive_loss": contrastive_loss})
    val_dist = best_model.predict([X_a_val, X_b_val], batch_size=BATCH_SIZE).ravel()
    thr, val_f1 = compute_threshold(y_val, val_dist)

    # Test evaluation
    test_dist = best_model.predict([X_a_test, X_b_test], batch_size=BATCH_SIZE).ravel()
    y_pred_test = (test_dist < thr).astype(int)

    metrics = {
        "val": {
            "best_threshold": thr,
            "best_val_f1": val_f1,
            "roc_auc": roc_auc_score(y_val, -val_dist),  # smaller distance => positive, use negative distance as score
            "pr_auc": average_precision_score(y_val, -val_dist),
            "accuracy": accuracy_score(y_val, (val_dist < thr).astype(int)),
        },
        "test": {
            "f1": f1_score(y_test, y_pred_test),
            "roc_auc": roc_auc_score(y_test, -test_dist),
            "pr_auc": average_precision_score(y_test, -test_dist),
            "accuracy": accuracy_score(y_test, y_pred_test),
        },
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epochs_trained": len(history.history["loss"]),
            "batch_size": BATCH_SIZE,
        }
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved evaluation summary to", SUMMARY_JSON)

if __name__ == "__main__":
    main()
