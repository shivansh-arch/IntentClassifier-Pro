"""
finetune/run_inference.py

Owner: Shivansh (Part 1)

Loads the trained LoRA adapter, runs inference on data/test.csv,
and writes finetune/predictions.csv in the shared schema format.

Run with: python finetune/run_inference.py
"""

import time
import pandas as pd
from shared.config import TEST_SPLIT, FINETUNE_PREDICTIONS, FINETUNE_ADAPTER_WEIGHTS
from shared.schema import PREDICTION_COLUMNS, MODEL_NAMES


def predict_batch(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Must return a DataFrame with exactly PREDICTION_COLUMNS.
    Measure latency_ms per single example (batch size 1) to keep the
    comparison against the baseline fair.
    """
    rows = []
    for _, row in test_df.iterrows():
        start = time.perf_counter()
        # TODO: tokenize row["text"], run through model, softmax -> predicted label + confidence
        predicted_label = None  # placeholder
        confidence = None       # placeholder
        latency_ms = (time.perf_counter() - start) * 1000

        rows.append({
            "text": row["text"],
            "true_label": row["label"],
            "predicted_label": predicted_label,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model_name": MODEL_NAMES["finetuned"],
        })
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SPLIT)
    predictions = predict_batch(test_df)
    predictions.to_csv(FINETUNE_PREDICTIONS, index=False)
    print(f"Saved {len(predictions)} predictions to {FINETUNE_PREDICTIONS}")
