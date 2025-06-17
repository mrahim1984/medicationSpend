
import streamlit as st
import pickle
import numpy as np

# Load model and brand_map
with open("medication_claim_model_rf.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

st.title("💊 Medication Claim Predictor")

# Input medication and year
med = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1)

# Predict
if st.button("Predict Next Year Claim"):
    X_input = np.array([[brand_map[med], year]])
    pred = model.predict(X_input)[0]
    st.success(f"Predicted Claims for {med} in {year+1}: {int(pred)}")
