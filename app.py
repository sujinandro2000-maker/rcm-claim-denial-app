import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(page_title="RCM Claim Denial Predictor", layout="wide")

st.title("🏥 RCM Claim Denial Prediction App")
st.write(
    "Predict whether a healthcare claim will be **Paid** or **Denied** before submission."
)

BASE_DIR = Path(__file__).resolve().parent


# 1. Load Model & Artifacts
@st.cache_resource
def load_artifacts():
  try:
    model_path = BASE_DIR / "claim_denial_model.joblib"
    fn_path = BASE_DIR / "feature_names.joblib"
    le_path = BASE_DIR / "label_encoders.joblib"
    ui_path = BASE_DIR / "ui_options.joblib"

    model = joblib.load(model_path)
    feature_names = joblib.load(fn_path)
    label_encoders = joblib.load(le_path)
    ui_options = joblib.load(ui_path)
    return model, feature_names, label_encoders, ui_options, None
  except Exception as e:
    return None, None, None, None, str(e)


model, feature_names, label_encoders, ui_options, err = load_artifacts()

# Display error on screen if files are missing or broken
if err:
  st.error(f"❌ Could not load model artifacts: {err}")
  st.info(
      "Please verify that all .joblib files are in the repository root"
      " directory."
  )
elif model is not None:
  st.success("Model artifacts loaded successfully!")
  st.markdown("---")
  st.subheader("📋 Enter Claim Details")

  input_data = {}
  col1, col2 = st.columns(2)

  if ui_options:
    for idx, (col_name, options) in enumerate(ui_options.items()):
      target_col = col1 if idx % 2 == 0 else col2
      with target_col:
        selected_val = st.selectbox(
            f"Select {col_name.replace('_', ' ').title()}",
            options,
            key=col_name,
        )
        if col_name in label_encoders:
          le = label_encoders[col_name]
          input_data[col_name] = le.transform([selected_val])[0]
        else:
          input_data[col_name] = selected_val

  with col2:
    prior_auth = st.toggle("Prior Authorization Required?", value=False)
    input_data["prior_auth_required"] = 1 if prior_auth else 0

  st.markdown("---")
  if st.button(
      "🔮 Predict Claim Outcome", type="primary", use_container_width=True
  ):
    input_df = pd.DataFrame([input_data])[feature_names]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Results")
    c1, c2 = st.columns(2)
    with c1:
      st.metric(label="Risk of Denial", value=f"{proba * 100:.1f}%")
    with c2:
      if proba >= 0.5:
        st.error(
            "🚨 **High Denial Risk** — Review claim details prior to submission."
        )
      else:
        st.success("✅ **Low Denial Risk** — Likely to be approved/paid.")
