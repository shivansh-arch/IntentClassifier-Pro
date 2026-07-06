"""
shared/schema.py

SINGLE SOURCE OF TRUTH for:
- the intent label list
- the predictions.csv column contract
- model name identifiers

Every part (finetune, baseline, evaluation, app) imports from HERE.
Nobody retypes label names or column names locally — that's how CSVs
stop matching each other across the team.
"""

# Full BANKING77 intent label list — paste the exact strings from the dataset
# object once (ds["train"].features["label"].names), then never touch this list again.
# Placeholder shown here — replace with the real 77 labels before anyone starts training.
INTENT_LABELS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    # ... continue with all 77 — do NOT hand-type these twice anywhere else in the repo
]

# Every predictions file (finetune AND baseline) must have exactly these columns,
# in this order, with these exact names.
PREDICTION_COLUMNS = [
    "text",            # the input sentence
    "true_label",      # ground-truth intent (must match a string in INTENT_LABELS)
    "predicted_label",  # model's predicted intent
    "confidence",      # float 0-1 if available, else None
    "latency_ms",      # float, inference time per single example
    "model_name",      # one of MODEL_NAMES values below
]

MODEL_NAMES = {
    "finetuned": "distilbert-lora-banking77",
    "baseline": "prompted-baseline",
}

# Bump this if the schema itself changes, so old CSVs can be identified as stale.
SCHEMA_VERSION = "1.0"
