"""
baseline/call_llm_baseline.py

Owner: Teammate 2 (Part 2)

Calls a large LLM API (Cerebras, gpt-oss-120b) in BATCHES with the
intent list + multiple test sentences, parses the answers, and writes
baseline/predictions.csv in the shared schema format.

Requires an API key — put it in a local .env (never commit it) and load
via python-dotenv locally, or st.secrets when deployed.

Run with: python -m baseline.call_llm_baseline
"""

import os
import time
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from shared.config import TEST_SPLIT, BASELINE_PREDICTIONS
from shared.schema import INTENT_LABELS, PREDICTION_COLUMNS, MODEL_NAMES

load_dotenv()
client = OpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1",
)

MODEL_NAME = "gpt-oss-120b"
BATCH_SIZE = 15
SECONDS_BETWEEN_CALLS = 13  # conservative, safe even if real RPM is only 5
MAX_RETRIES = 5
RATE_LIMIT_WAIT_SECONDS = 20

LABELS_BLOCK = "\n".join(INTENT_LABELS)


def build_batch_prompt(batch_texts: list[str]) -> str:
    numbered = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(batch_texts))
    return f"""You are an intent classifier. Here is the list of valid labels:
{LABELS_BLOCK}

Classify each numbered message below. Respond with ONLY a numbered list,
one label per line, matching the message numbers exactly. No explanations,
no extra text, no reasoning shown.

Messages:
{numbered}

Respond in exactly this format:
1. <label>
2. <label>
...
"""


def parse_batch_response(raw_text: str, expected_count: int) -> list[str]:
    lines = raw_text.strip().split("\n")
    labels = []
    for line in lines:
        match = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
        if match:
            label = match.group(1).strip()
            cleaned = label.lower().strip(".")
            matched_label = next((l for l in INTENT_LABELS if l.lower() == cleaned), None)
            labels.append(matched_label if matched_label else label)

    while len(labels) < expected_count:
        labels.append("PARSE_FAILED")

    return labels[:expected_count]


def classify_batch(batch_texts: list[str], max_retries: int = MAX_RETRIES) -> tuple[list[str], float]:
    prompt = build_batch_prompt(batch_texts)
    start = time.perf_counter()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )

            if hasattr(response, "usage") and response.usage:
                print(f"    [tokens used this call: {response.usage.total_tokens} "
                      f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})]")

            raw_output = response.choices[0].message.content
            latency_ms = (time.perf_counter() - start) * 1000
            labels = parse_batch_response(raw_output, len(batch_texts))
            return labels, latency_ms

        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit:
                print(f"Rate limited, waiting {RATE_LIMIT_WAIT_SECONDS}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            else:
                raise

    latency_ms = (time.perf_counter() - start) * 1000
    return ["RATE_LIMIT_FAILED"] * len(batch_texts), latency_ms


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SPLIT)
    # test_df = test_df.head(60)  # TEMP: sanity check on 4 batches, remove before full run

    rows = []
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].tolist()

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        batch_true = true_labels[batch_start:batch_start + BATCH_SIZE]

        predicted_labels, latency_ms = classify_batch(batch_texts)
        per_sentence_latency = latency_ms / len(batch_texts)

        for text, true_label, predicted_label in zip(batch_texts, batch_true, predicted_labels):
            rows.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": None,
                "latency_ms": per_sentence_latency,
                "model_name": MODEL_NAMES["baseline"],
            })
            print(f"'{text}' -> {predicted_label}")

        print(f"--- batch {batch_start // BATCH_SIZE + 1} done, {latency_ms:.0f}ms total ({per_sentence_latency:.0f}ms/sentence) ---")

        # Save incrementally after every batch, so a crash doesn't lose everything
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(BASELINE_PREDICTIONS, index=False)

        time.sleep(SECONDS_BETWEEN_CALLS)

    print(f"\nSaved {len(rows)} predictions to {BASELINE_PREDICTIONS}")