import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Page settings
st.set_page_config(
    page_title="Compost Maturity Predictor",
    page_icon="🌱",
    layout="centered"
)

# Load trained XGBoost model
model = XGBClassifier()
model.load_model("xgboost_compost_model.json")

# Load label encoder and feature names
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("🌱 Compost Maturity Predictor")

st.write(
    "Enter the compost parameters below to predict "
    "the maturity status of the compost."
)

st.divider()

# Input parameters
day = st.number_input(
    "Day",
    min_value=0.0,
    value=0.0
)

temperature = st.number_input(
    "Temperature",
    value=0.0
)

mc = st.number_input(
    "Moisture Content (%)",
    min_value=0.0,
    value=0.0
)

ph = st.number_input(
    "pH",
    min_value=0.0,
    value=0.0
)

cn_ratio = st.number_input(
    "C/N Ratio",
    min_value=0.0,
    value=0.0
)

ammonia = st.number_input(
    "Ammonia (mg/kg)",
    min_value=0.0,
    value=0.0
)

nitrate = st.number_input(
    "Nitrate (mg/kg)",
    min_value=0.0,
    value=0.0
)

tn = st.number_input(
    "TN (%)",
    min_value=0.0,
    value=0.0
)

toc = st.number_input(
    "TOC (%)",
    min_value=0.0,
    value=0.0
)

ec = st.number_input(
    "EC (ms/cm)",
    min_value=0.0,
    value=0.0
)

om = st.number_input(
    "OM (%)",
    min_value=0.0,
    value=0.0
)

t_value = st.number_input(
    "T Value",
    min_value=0.0,
    value=0.0
)

gi = st.number_input(
    "GI (%)",
    min_value=0.0,
    value=0.0
)

# Create input dataframe
input_data = pd.DataFrame([[
    day,
    temperature,
    mc,
    ph,
    cn_ratio,
    ammonia,
    nitrate,
    tn,
    toc,
    ec,
    om,
    t_value,
    gi
]], columns=feature_names)

# Prediction button
if st.button("🔍 Predict Maturity"):

    # Check if all inputs are zero
    if all(value == 0 for value in input_data.iloc[0]):

        st.warning(
            "⚠️ Please enter the compost parameter values before predicting."
        )

    else:

        prediction = model.predict(input_data)

        predicted_class = label_encoder.inverse_transform(
            prediction
        )[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = probabilities.max() * 100

        st.success(
            f"Predicted Maturity Status: **{predicted_class}**"
        )

        st.info(
            f"Prediction Confidence: **{confidence:.2f}%**"
        )

        # Probability table
        probability_df = pd.DataFrame({
            "Maturity Status": label_encoder.classes_,
            "Probability (%)": probabilities * 100
        })

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df,
            hide_index=True
        )