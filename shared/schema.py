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
INTENT_LABELS = ['activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support', 'automatic_top_up', 'balance_not_updated_after_bank_transfer', 'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed', 'cancel_transfer', 'card_about_to_expire', 'card_acceptance', 'card_arrival', 'card_delivery_estimate', 'card_linking', 'card_not_working', 'card_payment_fee_charged', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'card_swallowed', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised', 'change_pin', 'compromised_card', 'contactless_not_working', 'country_support', 'declined_card_payment', 'declined_cash_withdrawal', 'declined_transfer', 'direct_debit_payment_not_recognised', 'disposable_card_limits', 'edit_personal_details', 'exchange_charge', 'exchange_rate', 'exchange_via_app', 'extra_charge_on_statement', 'failed_transfer', 'fiat_currency_support', 'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card', 'getting_virtual_card', 'lost_or_stolen_card', 'lost_or_stolen_phone', 'order_physical_card', 'passcode_forgotten', 'pending_card_payment', 'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked', 'receiving_money', 'Refund_not_showing_up', 'request_refund', 'reverted_card_payment?', 'supported_cards_and_currencies', 'terminate_account', 'top_up_by_bank_transfer_charge', 'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits', 'top_up_reverted', 'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged', 'transfer_into_account', 'transfer_not_received_by_recipient', 'transfer_timing', 'unable_to_verify_identity', 'verify_my_identity', 'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working', 'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received', 'wrong_exchange_rate_for_cash_withdrawal']

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
