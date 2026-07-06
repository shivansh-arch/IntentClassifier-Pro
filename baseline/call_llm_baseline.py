"""
baseline/call_llm_baseline.py

Owner: Teammate 2 (Part 2)

Calls a large LLM API with the intent list + test sentence, parses its
answer, and writes baseline/predictions.csv in the shared schema format.

Requires an API key — put it in a local .env (never commit it) and load
via python-dotenv locally, or st.secrets when deployed.

Run with: python baseline/call_llm_baseline.py
"""

import time
import pandas as pd
from shared.config import TEST_SPLIT, BASELINE_PREDICTIONS
from shared.schema import INTENT_LABELS, PREDICTION_COLUMNS, MODEL_NAMES

PROMPT_TEMPLATE = """You are an intent classifier. Given the message below, respond with
exactly one label from this list, and nothing else:
{labels}

Message: "{text}"
Label:"""


def predict_intent(text: str) -> tuple[str, float, float]:
    """
    Returns (predicted_label, confidence, latency_ms).
    confidence may be None if the API doesn't return a usable probability —
    don't fabricate a precise-looking number if you don't have one.
    """
    prompt = PROMPT_TEMPLATE.format(labels="\n".join(INTENT_LABELS), text=text)
    start = time.perf_counter()
    # TODO: call the LLM API with `prompt`, parse the returned label string
    predicted_label = None  # placeholder
    confidence = None       # placeholder
    latency_ms = (time.perf_counter() - start) * 1000
    return predicted_label, confidence, latency_ms


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SPLIT)
    rows = []
    for _, row in test_df.iterrows():
        predicted_label, confidence, latency_ms = predict_intent(row["text"])
        rows.append({
            "text": row["text"],
            "true_label": row["label"],
            "predicted_label": predicted_label,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model_name": MODEL_NAMES["baseline"],
        })
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    predictions.to_csv(BASELINE_PREDICTIONS, index=False)
    print(f"Saved {len(predictions)} predictions to {BASELINE_PREDICTIONS}")
