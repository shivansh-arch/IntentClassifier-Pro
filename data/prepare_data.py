"""
data/prepare_data.py

Loads BANKING77, creates stratified train/val split (test split is used as-is),
saves all three as CSVs matching the columns other scripts expect: text, label.

Run with: python data/prepare_data.py
(This is a shared prerequisite — whoever gets to it first on Day 1-2 runs it
and commits the CSVs so the rest of the team isn't blocked.)
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from shared.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

if __name__ == "__main__":
    ds = load_dataset("PolyAI/banking77")

    train_full = ds["train"].to_pandas()
    test = ds["test"].to_pandas()

    label_names = ds["train"].features["label"].names
    train_full["label"] = train_full["label"].map(lambda i: label_names[i])
    test["label"] = test["label"].map(lambda i: label_names[i])

    train, val = train_test_split(
        train_full, test_size=0.1, stratify=train_full["label"], random_state=42
    )

    train.to_csv(TRAIN_SPLIT, index=False)
    val.to_csv(VAL_SPLIT, index=False)
    test.to_csv(TEST_SPLIT, index=False)

    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print("IMPORTANT: copy label_names into shared/schema.py -> INTENT_LABELS, exactly as printed below:")
    print(label_names)
