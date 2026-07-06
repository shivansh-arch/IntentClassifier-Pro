"""
shared/config.py

Central place for every file path used across the project.
Nobody writes a raw string path in their own script — import from here.
This is what stops "works on my machine" path bugs when the repo
moves to someone else's machine, Colab, or a Streamlit Cloud container.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent  # repo root, regardless of who runs it or from where

# Data
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_SPLIT = DATA_DIR / "train.csv"
VAL_SPLIT = DATA_DIR / "val.csv"
TEST_SPLIT = DATA_DIR / "test.csv"

# Part 1 outputs
FINETUNE_DIR = ROOT / "finetune"
FINETUNE_PREDICTIONS = FINETUNE_DIR / "predictions.csv"
FINETUNE_ADAPTER_WEIGHTS = FINETUNE_DIR / "lora_adapter"  # saved PEFT adapter folder

# Part 2 outputs
BASELINE_DIR = ROOT / "baseline"
BASELINE_PREDICTIONS = BASELINE_DIR / "predictions.csv"

# Part 3 outputs
EVAL_DIR = ROOT / "evaluation"
CONFUSION_MATRIX_PNG = EVAL_DIR / "confusion_matrix.png"
TOP_CONFUSIONS_CSV = EVAL_DIR / "top_confusions.csv"
METRICS_SUMMARY_JSON = EVAL_DIR / "metrics_summary.json"

# App
APP_DIR = ROOT / "app"
