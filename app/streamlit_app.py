"""
app/streamlit_app.py

Owner: Teammate 2 (Part 2)

Live demo: user types a sentence, sees both models' predictions side by
side. Also shows the static evaluation dashboard (accuracy/latency table,
confusion matrix) from evaluation/ outputs.

Run locally with: streamlit run app/streamlit_app.py
API keys: use st.secrets["API_KEY"] — configure in Streamlit Cloud's
Secrets manager after deploying, never hardcode here.
"""

import streamlit as st
import pandas as pd
import json

from shared.config import FINETUNE_PREDICTIONS, BASELINE_PREDICTIONS, METRICS_SUMMARY_JSON

st.set_page_config(page_title="IntentClassifier Pro", layout="wide")
st.title("IntentClassifier Pro — Tuned vs Prompted")

tab1, tab2 = st.tabs(["Live Demo", "Results Dashboard"])

with tab1:
    user_text = st.text_input("Type a banking-related message:")
    if user_text:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fine-tuned model (LoRA)")
            # TODO: load the trained model once at startup (cache with st.cache_resource),
            # run inference on user_text, display predicted label + confidence + latency
            st.write("prediction goes here")
        with col2:
            st.subheader("Prompting baseline")
            # TODO: call the same baseline function from baseline/call_llm_baseline.py
            st.write("prediction goes here")

with tab2:
    if METRICS_SUMMARY_JSON.exists():
        with open(METRICS_SUMMARY_JSON) as f:
            summary = json.load(f)
        st.json(summary)
    else:
        st.info("Run evaluation/build_confusion_matrix.py first to generate results.")
