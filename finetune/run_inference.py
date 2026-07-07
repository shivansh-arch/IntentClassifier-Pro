"""
finetune/run_inference.py
"""

import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from shared.config import TEST_SPLIT, FINETUNE_PREDICTIONS, FINETUNE_ADAPTER_WEIGHTS
from shared.schema import INTENT_LABELS, PREDICTION_COLUMNS, MODEL_NAMES

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(INTENT_LABELS)
)
model = PeftModel.from_pretrained(base_model, FINETUNE_ADAPTER_WEIGHTS)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_batch(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            start = time.perf_counter()
            encoding = tokenizer(
                row["text"], truncation=True, padding="max_length",
                max_length=64, return_tensors="pt"
            ).to(device)

            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)

            latency_ms = (time.perf_counter() - start) * 1000

            rows.append({
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": INTENT_LABELS[pred_idx.item()],
                "confidence": confidence.item(),
                "latency_ms": latency_ms,
                "model_name": MODEL_NAMES["finetuned"],
            })
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SPLIT)
    predictions = predict_batch(test_df)
    predictions.to_csv(FINETUNE_PREDICTIONS, index=False)
    print(f"Saved {len(predictions)} predictions to {FINETUNE_PREDICTIONS}")
    print(f"Overall accuracy: {(predictions['true_label'] == predictions['predicted_label']).mean():.4f}")