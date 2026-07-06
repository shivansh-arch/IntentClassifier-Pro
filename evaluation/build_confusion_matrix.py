"""
evaluation/build_confusion_matrix.py

Owner: whoever's turn it is to run eval (usually driven by Part 3's needs,
code itself can be written by you or teammate 2 — Part 3 reads the OUTPUT
of this file, doesn't need to write it).

Reads finetune/predictions.csv and baseline/predictions.csv, computes
accuracy, latency comparison, and top confused-intent pairs.

IMPORTANT: if real predictions.csv files don't exist yet, this generates
dummy random predictions so Part 3 can build their analysis workflow
without waiting idle. Swap to real files the moment they exist.

Run with: python evaluation/build_confusion_matrix.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from shared.config import (
    FINETUNE_PREDICTIONS, BASELINE_PREDICTIONS, TEST_SPLIT,
    TOP_CONFUSIONS_CSV, METRICS_SUMMARY_JSON,
)
from shared.schema import INTENT_LABELS, MODEL_NAMES


def load_or_dummy(path, model_name):
    if path.exists():
        return pd.read_csv(path)
    print(f"[warning] {path} not found — generating dummy predictions for now.")
    test_df = pd.read_csv(TEST_SPLIT)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "text": test_df["text"],
        "true_label": test_df["label"],
        "predicted_label": rng.choice(INTENT_LABELS, size=len(test_df)),
        "confidence": rng.uniform(0.5, 1.0, size=len(test_df)),
        "latency_ms": rng.uniform(5, 300, size=len(test_df)),
        "model_name": model_name,
    })


def top_confused_pairs(true_labels, pred_labels, labels, top_n=10):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    np.fill_diagonal(cm, 0)
    flat_idx = np.argsort(cm.ravel())[-top_n:][::-1]
    pairs = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, cm.shape)
        if cm[i, j] > 0:
            pairs.append({"true_label": labels[i], "confused_with": labels[j], "count": int(cm[i, j])})
    return pd.DataFrame(pairs)


if __name__ == "__main__":
    ft = load_or_dummy(FINETUNE_PREDICTIONS, MODEL_NAMES["finetuned"])
    bl = load_or_dummy(BASELINE_PREDICTIONS, MODEL_NAMES["baseline"])

    summary = {
        "finetuned_accuracy": accuracy_score(ft["true_label"], ft["predicted_label"]),
        "baseline_accuracy": accuracy_score(bl["true_label"], bl["predicted_label"]),
        "finetuned_avg_latency_ms": float(ft["latency_ms"].mean()),
        "baseline_avg_latency_ms": float(bl["latency_ms"].mean()),
    }
    print(summary)

    confusions = top_confused_pairs(ft["true_label"], ft["predicted_label"], INTENT_LABELS)
    confusions.to_csv(TOP_CONFUSIONS_CSV, index=False)

    import json
    with open(METRICS_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved top confusions to {TOP_CONFUSIONS_CSV}")
    print(f"Saved metrics summary to {METRICS_SUMMARY_JSON}")
