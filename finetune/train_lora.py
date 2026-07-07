"""
finetune/train_lora.py

Owner: Shivansh (Part 1)

Pipeline:
1. Load data/train.csv and data/val.csv
2. Load base model (DistilBERT) + attach LoRA adapters via PEFT
3. Train, tracking val loss/accuracy per epoch
4. Save adapter weights to finetune/lora_adapter/

Run with: python -m finetune.train_lora
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from shared.config import TRAIN_SPLIT, VAL_SPLIT, FINETUNE_ADAPTER_WEIGHTS
from shared.schema import INTENT_LABELS

label_to_idx = {label: idx for idx, label in enumerate(INTENT_LABELS)}

# --- Load base model + tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(INTENT_LABELS)
)

# --- Attach LoRA ---
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --- Dataset ---
class IntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        label_idx = label_to_idx[row["label"]]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


# TODO: wrap IntentDataset(TRAIN_SPLIT, ...) and IntentDataset(VAL_SPLIT, ...) in DataLoaders
# TODO: manual training loop — forward pass, CrossEntropyLoss, backward, optimizer.step(), per epoch
# TODO: per-epoch validation loop (no gradient updates, just measure val loss/accuracy)
# TODO: model.save_pretrained(FINETUNE_ADAPTER_WEIGHTS)
if __name__ == "__main__":
    train_dataset = IntentDataset(TRAIN_SPLIT, tokenizer)
    val_dataset = IntentDataset(VAL_SPLIT, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"          val_loss={avg_val_loss:.4f}, val_accuracy={val_accuracy:.4f}")

    model.save_pretrained(FINETUNE_ADAPTER_WEIGHTS)
    print(f"Saved adapter to {FINETUNE_ADAPTER_WEIGHTS}")