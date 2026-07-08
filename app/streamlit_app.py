"""
app/streamlit_app.py

Owner: Teammate 2 & Shivansh (Part 2)

Live demo: user types a sentence, sees both models' predictions side by side.
Also shows the static evaluation dashboard (accuracy/latency table, confusion matrix)
from evaluation/ outputs.

Run locally with: streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

# Add root folder to sys.path to ensure absolute imports work correctly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from shared.config import (
    FINETUNE_ADAPTER_WEIGHTS,
    METRICS_SUMMARY_JSON,
    CONFUSION_MATRIX_PNG,
    TOP_CONFUSIONS_CSV
)
from shared.schema import INTENT_LABELS
from baseline.call_llm_baseline import classify_batch

# --- Page Configurations ---
st.set_page_config(
    page_title="IntentClassifier Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0E1117;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Configuration ---
st.sidebar.title("🤖 Configuration")
st.sidebar.markdown("---")
st.sidebar.subheader("API Access")

# Retrieve API keys from env, secrets, or manual input
env_key = os.environ.get("CEREBRAS_API_KEY", "")
secrets_key = ""
try:
    if hasattr(st, "secrets") and "CEREBRAS_API_KEY" in st.secrets:
        secrets_key = st.secrets["CEREBRAS_API_KEY"]
except Exception:
    pass

default_key = secrets_key or env_key

api_key_input = st.sidebar.text_input(
    "Cerebras API Key",
    value=default_key,
    type="password",
    help="Enter your Cerebras API key to run inference on the baseline model."
)

active_api_key = api_key_input.strip()

if not active_api_key:
    st.sidebar.warning("⚠️ Cerebras API Key is missing. Baseline LLM inference will not run.")
else:
    st.sidebar.success("🔑 API Key configured.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **IntentClassifier Pro** comparisons:
    * **Fine-Tuned**: DistilBERT + LoRA
    * **Baseline**: Prompted gpt-oss-120b
    """
)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_finetuned_model():
    """Loads tokenizer and PEFT model on the GPU (if available) or CPU."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(INTENT_LABELS)
    )
    
    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, FINETUNE_ADAPTER_WEIGHTS)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device

# --- Main Page Layout ---
st.title("IntentClassifier Pro — Tuned vs Prompted")
st.markdown("Compares a fine-tuned LoRA DistilBERT model against an LLM-prompted baseline for intent classification on the BANKING77 dataset.")

tab1, tab2 = st.tabs(["🚀 Live Demo", "📊 Results Dashboard"])

# --- TAB 1: Live Demo ---
with tab1:
    st.subheader("Interactive Single-Sentence Testing")
    st.markdown("Type a banking-related sentence below to run classification on both models in real-time.")

    user_text = st.text_input(
        "User Message:",
        placeholder="e.g., I still haven't received my new card, when will it arrive?",
        key="user_text_input"
    )

    if user_text:
        # Load and run fine-tuned model
        with st.spinner("Running fine-tuned model..."):
            try:
                tokenizer, ft_model, device = load_finetuned_model()
                
                start_time = time.perf_counter()
                
                # Tokenize and predict
                encoding = tokenizer(
                    user_text, truncation=True, padding="max_length",
                    max_length=64, return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = ft_model(**encoding)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence, pred_idx = torch.max(probs, dim=-1)
                    
                ft_latency = (time.perf_counter() - start_time) * 1000
                ft_label = INTENT_LABELS[pred_idx.item()]
                ft_confidence_pct = confidence.item() * 100
                ft_error = None
            except Exception as e:
                ft_error = str(e)
                ft_label, ft_confidence_pct, ft_latency = "Error", 0.0, 0.0

        # Run baseline model
        with st.spinner("Running prompted baseline model..."):
            if not active_api_key:
                bl_label, bl_latency = "API KEY MISSING", 0.0
                bl_error = "Please provide your Cerebras API key in the sidebar to run baseline classification."
            else:
                try:
                    predicted_labels, bl_latency = classify_batch([user_text], api_key=active_api_key)
                    bl_label = predicted_labels[0]
                    bl_error = None
                    if bl_label in ["CLIENT_INIT_FAILED", "RATE_LIMIT_FAILED", "PARSE_FAILED"]:
                        bl_error = f"Baseline API returned error status: {bl_label}"
                except Exception as e:
                    bl_error = str(e)
                    bl_label, bl_latency = "Error", 0.0

        # Display side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚡ Fine-Tuned Model (LoRA)")
            with st.container(border=True):
                if ft_error:
                    st.error(f"Failed to run fine-tuned model: {ft_error}")
                else:
                    st.markdown(f"**Predicted Intent:** `{ft_label}`")
                    st.markdown(f"**Confidence:** `{ft_confidence_pct:.2f}%`")
                    st.markdown(f"**Latency:** `{ft_latency:.2f} ms`")

        with col2:
            st.markdown("### 🧠 Prompted Baseline")
            with st.container(border=True):
                if bl_error:
                    st.error(f"Failed to run baseline model: {bl_error}")
                else:
                    st.markdown(f"**Predicted Intent:** `{bl_label}`")
                    st.markdown("**Confidence:** `N/A` (Few-shot prompting)")
                    st.markdown(f"**Latency:** `{bl_latency:.2f} ms`")

# --- TAB 2: Results Dashboard ---
with tab2:
    st.subheader("Offline Model Performance Dashboard")
    st.markdown("Static evaluation metrics computed over the full BANKING77 test set split.")

    if METRICS_SUMMARY_JSON.exists():
        with open(METRICS_SUMMARY_JSON) as f:
            summary = json.load(f)

        # Metrics cards
        st.markdown("#### 📈 Summary Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(
                label="Fine-tuned Accuracy",
                value=f"{summary['finetuned_accuracy'] * 100:.2f}%",
                delta=f"{(summary['finetuned_accuracy'] - summary['baseline_accuracy']) * 100:+.2f}% vs Baseline"
            )
        with m_col2:
            st.metric(
                label="Baseline Accuracy",
                value=f"{summary['baseline_accuracy'] * 100:.2f}%"
            )
        with m_col3:
            st.metric(
                label="Fine-tuned Avg Latency",
                value=f"{summary['finetuned_avg_latency_ms']:.1f} ms",
                delta=f"{summary['finetuned_avg_latency_ms'] - summary['baseline_avg_latency_ms']:.1f} ms",
                delta_color="inverse"
            )
        with m_col4:
            st.metric(
                label="Baseline Avg Latency",
                value=f"{summary['baseline_avg_latency_ms']:.1f} ms"
            )

        st.markdown("---")

        # Visualizations (Confusion Matrix and Top Confusions)
        d_col1, d_col2 = st.columns([3, 2])

        with d_col1:
            st.markdown("#### 📉 Confusion Matrix")
            st.markdown("Legible 20x20 confusion matrix for the most common intents in the dataset.")
            if CONFUSION_MATRIX_PNG.exists():
                st.image(str(CONFUSION_MATRIX_PNG), use_column_width=True)
            else:
                st.info("Confusion matrix image not found. Run evaluation/build_confusion_matrix.py to generate it.")

        with d_col2:
            st.markdown("#### 🔍 Top 10 Intent Confusions")
            st.markdown("The pairs of intents that the fine-tuned model confused most frequently.")
            if TOP_CONFUSIONS_CSV.exists():
                try:
                    confusions_df = pd.read_csv(TOP_CONFUSIONS_CSV)
                    confusions_df.columns = ["True Intent", "Confused With", "Error Count"]
                    st.dataframe(confusions_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error loading confusion pairs: {e}")
            else:
                st.info("Top confusions CSV not found.")
    else:
        st.info("Evaluation results not found. Run the pipelines to generate them.")

