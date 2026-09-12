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
from difflib import SequenceMatcher

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
    BASELINE_PREDICTIONS,
    METRICS_SUMMARY_JSON,
    CONFUSION_MATRIX_PNG,
    TOP_CONFUSIONS_CSV
)

from shared.schema import INTENT_LABELS


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

st.sidebar.subheader("Baseline Status")
st.sidebar.success("✅ Using pre-generated baseline predictions")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **IntentClassifier Pro** comparisons:
    * **Fine-Tuned**: DistilBERT + LoRA
    * **Baseline**: Pre-generated LLM predictions
    """
)


# --- Model Loading (Cached) ---
@st.cache_resource
def load_finetuned_model():
    """Loads tokenizer and PEFT model on GPU if available, otherwise CPU."""

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(INTENT_LABELS)
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(FINETUNE_ADAPTER_WEIGHTS)
    )

    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    return tokenizer, model, device


# --- Baseline Prediction Lookup ---
def get_baseline_prediction(user_text, baseline_df):
    """
    Finds the closest pre-generated baseline prediction
    for the user's input sentence.

    First tries an exact match.
    If no exact match exists, uses SequenceMatcher
    to find a sufficiently similar sentence.
    """

    normalized_input = user_text.strip().lower()

    # -------------------------------------------------
    # 1. Try exact match first
    # -------------------------------------------------
    exact_matches = baseline_df[
        baseline_df["text"]
        .astype(str)
        .str.strip()
        .str.lower()
        == normalized_input
    ]

    if not exact_matches.empty:
        row = exact_matches.iloc[0]

        return (
            row["predicted_label"],
            float(row["latency_ms"]),
            None,
            1.0
        )

    # -------------------------------------------------
    # 2. Try similarity matching
    # -------------------------------------------------
    best_match = None
    best_score = 0.0

    for _, row in baseline_df.iterrows():

        existing_text = (
            str(row["text"])
            .strip()
            .lower()
        )

        score = SequenceMatcher(
            None,
            normalized_input,
            existing_text
        ).ratio()

        if score > best_score:
            best_score = score
            best_match = row

    # -------------------------------------------------
    # 3. Accept sufficiently similar sentence
    # -------------------------------------------------
    if best_match is not None and best_score >= 0.50:

        return (
            best_match["predicted_label"],
            float(best_match["latency_ms"]),
            None,
            best_score
        )

    # -------------------------------------------------
    # 4. No suitable prediction found
    # -------------------------------------------------
    return (
        "Not Available",
        0.0,
        "No sufficiently similar pre-generated baseline prediction exists for this sentence.",
        best_score
    )


# --- Main Page Layout ---
st.title("IntentClassifier Pro — Tuned vs Prompted")

st.markdown(
    "Compares a fine-tuned LoRA DistilBERT model against an "
    "LLM-prompted baseline for intent classification on the BANKING77 dataset."
)


tab1, tab2 = st.tabs(
    ["🚀 Live Demo", "📊 Results Dashboard"]
)


# =====================================================
# TAB 1: LIVE DEMO
# =====================================================

with tab1:

    st.subheader("Interactive Single-Sentence Testing")

    st.markdown(
        "Type a banking-related sentence below to compare "
        "the fine-tuned model with a pre-generated baseline prediction."
    )

    user_text = st.text_input(
        "User Message:",
        placeholder=(
            "e.g., I still haven't received my new card, "
            "when will it arrive?"
        ),
        key="user_text_input"
    )

    if user_text:

        # =================================================
        # Fine-Tuned Model
        # =================================================

        with st.spinner("Running fine-tuned model..."):

            try:

                tokenizer, ft_model, device = load_finetuned_model()

                start_time = time.perf_counter()

                # Tokenize input
                encoding = tokenizer(
                    user_text,
                    truncation=True,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt"
                ).to(device)

                # Prediction
                with torch.no_grad():

                    outputs = ft_model(**encoding)

                    probs = torch.softmax(
                        outputs.logits,
                        dim=-1
                    )

                    confidence, pred_idx = torch.max(
                        probs,
                        dim=-1
                    )

                ft_latency = (
                    time.perf_counter() - start_time
                ) * 1000

                ft_label = INTENT_LABELS[
                    pred_idx.item()
                ]

                ft_confidence_pct = (
                    confidence.item() * 100
                )

                ft_error = None

            except Exception as e:

                ft_error = str(e)

                ft_label = "Error"
                ft_confidence_pct = 0.0
                ft_latency = 0.0


        # =================================================
        # Pre-generated Baseline Model
        # =================================================

        with st.spinner(
            "Loading pre-generated baseline prediction..."
        ):

            try:

                baseline_df = pd.read_csv(
                    BASELINE_PREDICTIONS
                )

                (
                    bl_label,
                    bl_latency,
                    bl_error,
                    match_score
                ) = get_baseline_prediction(
                    user_text,
                    baseline_df
                )

            except Exception as e:

                bl_label = "Error"
                bl_latency = 0.0
                bl_error = str(e)
                match_score = 0.0


        # =================================================
        # Display Side-by-Side Comparison
        # =================================================

        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # Fine-Tuned Model
        # -------------------------------------------------

        with col1:

            st.markdown(
                "### ⚡ Fine-Tuned Model (LoRA)"
            )

            with st.container(border=True):

                if ft_error:

                    st.error(
                        f"Failed to run fine-tuned model: "
                        f"{ft_error}"
                    )

                else:

                    st.markdown(
                        f"**Predicted Intent:** "
                        f"`{ft_label}`"
                    )

                    st.markdown(
                        f"**Confidence:** "
                        f"`{ft_confidence_pct:.2f}%`"
                    )

                    st.markdown(
                        f"**Latency:** "
                        f"`{ft_latency:.2f} ms`"
                    )


        # -------------------------------------------------
        # Prompted Baseline
        # -------------------------------------------------

        with col2:

            st.markdown(
                "### 🧠 Prompted Baseline"
            )

            with st.container(border=True):

                if bl_error:

                    st.error(
                        f"Baseline prediction unavailable: "
                        f"{bl_error}"
                    )

                else:

                    st.markdown(
                        f"**Predicted Intent:** "
                        f"`{bl_label}`"
                    )

                    st.markdown(
                        "**Confidence:** `N/A` "
                        "(Pre-generated LLM prediction)"
                    )

                    st.markdown(
                        f"**Latency:** "
                        f"`{bl_latency:.2f} ms`"
                    )

                    if match_score < 1.0:

                        st.caption(
                            f"Matched similar test sentence "
                            f"(similarity: {match_score * 100:.1f}%)"
                        )


# =====================================================
# TAB 2: RESULTS DASHBOARD
# =====================================================

with tab2:

    st.subheader(
        "Offline Model Performance Dashboard"
    )

    st.markdown(
        "Static evaluation metrics computed over the "
        "full BANKING77 test set split."
    )


    if METRICS_SUMMARY_JSON.exists():

        with open(METRICS_SUMMARY_JSON) as f:

            summary = json.load(f)


        # =================================================
        # Metrics Cards
        # =================================================

        st.markdown("#### 📈 Summary Metrics")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)


        # Fine-tuned accuracy
        with m_col1:

            st.metric(
                label="Fine-tuned Accuracy",
                value=(
                    f"{summary['finetuned_accuracy'] * 100:.2f}%"
                ),
                delta=(
                    f"{(summary['finetuned_accuracy'] - summary['baseline_accuracy']) * 100:+.2f}% vs Baseline"
                )
            )


        # Baseline accuracy
        with m_col2:

            st.metric(
                label="Baseline Accuracy",
                value=(
                    f"{summary['baseline_accuracy'] * 100:.2f}%"
                )
            )


        # Fine-tuned latency
        with m_col3:

            st.metric(
                label="Fine-tuned Avg Latency",
                value=(
                    f"{summary['finetuned_avg_latency_ms']:.1f} ms"
                ),
                delta=(
                    f"{summary['finetuned_avg_latency_ms'] - summary['baseline_avg_latency_ms']:.1f} ms"
                ),
                delta_color="inverse"
            )


        # Baseline latency
        with m_col4:

            st.metric(
                label="Baseline Avg Latency",
                value=(
                    f"{summary['baseline_avg_latency_ms']:.1f} ms"
                )
            )


        st.markdown("---")


        # =================================================
        # Confusion Matrix + Top Confusions
        # =================================================

        d_col1, d_col2 = st.columns([3, 2])


        # Confusion matrix
        with d_col1:

            st.markdown(
                "#### 📉 Confusion Matrix"
            )

            st.markdown(
                "Legible 20x20 confusion matrix for "
                "the most common intents in the dataset."
            )

            if CONFUSION_MATRIX_PNG.exists():

                st.image(
                    str(CONFUSION_MATRIX_PNG),
                    width="stretch"
                )

            else:

                st.info(
                    "Confusion matrix image not found. "
                    "Run evaluation/build_confusion_matrix.py "
                    "to generate it."
                )


        # Top confusions
        with d_col2:

            st.markdown(
                "#### 🔍 Top 10 Intent Confusions"
            )

            st.markdown(
                "The pairs of intents that the fine-tuned "
                "model confused most frequently."
            )

            if TOP_CONFUSIONS_CSV.exists():

                try:

                    confusions_df = pd.read_csv(
                        TOP_CONFUSIONS_CSV
                    )

                    confusions_df.columns = [
                        "True Intent",
                        "Confused With",
                        "Error Count"
                    ]

                    st.dataframe(
                        confusions_df,
                        use_container_width=True,
                        hide_index=True
                    )

                except Exception as e:

                    st.error(
                        f"Error loading confusion pairs: {e}"
                    )

            else:

                st.info(
                    "Top confusions CSV not found."
                )


    else:

        st.info(
            "Evaluation results not found. "
            "Run the pipelines to generate them."
        )