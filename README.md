# IntentClassifier Pro — Tuned vs Prompted

Compares a LoRA-fine-tuned DistilBERT against a prompting baseline for intent
classification on BANKING77.

## Team & Ownership

| Part | Owner | Folder |
|---|---|---|
| 1 — Fine-tuning | Shivansh | `finetune/` |
| 2 — Baseline + App | Teammate 2 | `baseline/`, `app/` |
| 3 — Analysis + Report | Teammate 3 | reads `evaluation/` outputs |

## Setup (everyone runs this first)

```bash
git clone <repo-url>
cd intentclassifier-pro
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your own API key locally
```

## Day 1 — before anyone writes real code

1. Run `python data/prepare_data.py` once (whoever gets to it first).
2. Copy the printed label list into `shared/schema.py` → `INTENT_LABELS`. Do this exactly once — every other file imports from there.
3. Confirm everyone can import `shared.config` and `shared.schema` without errors.

## Folder structure

```
intentclassifier-pro/
├── shared/              # schema.py + config.py — the contract everyone imports, edited rarely
├── data/                # prepare_data.py + generated train/val/test CSVs
├── finetune/            # Part 1 — LoRA training + inference
├── baseline/            # Part 2 — prompting baseline
├── evaluation/          # Part 3 reads outputs here — confusion matrix, metrics
├── app/                 # Streamlit demo (Part 2)
├── requirements.txt     # combined, used for deployment
└── .env.example         # documents required secrets without exposing real ones
```

## Running each part

```bash
python data/prepare_data.py                 # once, Day 1
python finetune/train_lora.py                 # Part 1
python finetune/run_inference.py              # Part 1 -> finetune/predictions.csv
python baseline/call_llm_baseline.py          # Part 2 -> baseline/predictions.csv
python evaluation/build_confusion_matrix.py   # Part 3 input -> evaluation/*.csv, *.json
streamlit run app/streamlit_app.py            # Part 2 -> live demo
```

## Rules that keep this from breaking

- Never hand-type a label name or CSV column — import from `shared/schema.py`.
- Never hardcode a file path — import from `shared/config.py`.
- Never commit `.env` or real API keys — use `.env.example` as the template, Streamlit Cloud Secrets when deployed.
- `evaluation/build_confusion_matrix.py` auto-generates dummy predictions if the real CSVs don't exist yet, so Part 3 is never blocked waiting on Parts 1/2.

See `IntentClassifier_Pro_Pipeline_Deep_Dive.md` for the in-depth explanation of what each stage is actually doing.
