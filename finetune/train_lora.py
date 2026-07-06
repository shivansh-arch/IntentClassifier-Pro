"""
finetune/train_lora.py

Owner: Shivansh (Part 1)

Pipeline:
1. Load data/train.csv and data/val.csv
2. Load base model (DistilBERT) + attach LoRA adapters via PEFT
3. Train, tracking val loss/accuracy per epoch
4. Save adapter weights to finetune/lora_adapter/

Run with: python finetune/train_lora.py
"""

from shared.config import TRAIN_SPLIT, VAL_SPLIT, FINETUNE_ADAPTER_WEIGHTS
from shared.schema import INTENT_LABELS

# TODO: load base model + tokenizer (distilbert-base-uncased)
# TODO: build LoraConfig (task_type=SEQ_CLS, target_modules=["q_lin", "v_lin"])
# TODO: wrap model with get_peft_model()
# TODO: build Dataset/DataLoader from TRAIN_SPLIT / VAL_SPLIT
# TODO: training loop with per-epoch validation
# TODO: model.save_pretrained(FINETUNE_ADAPTER_WEIGHTS)

if __name__ == "__main__":
    raise NotImplementedError("Fill in the training loop — see Pipeline Deep Dive doc, section 2.")
