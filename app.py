
import streamlit as st
import pickle
import numpy as np

# Load model
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# UI
st.title("💊 Medication Claim Predictor")
meds = list(brand_map.keys())
selected_med = st.selectbox("Select Medication", meds)
year = st.slider("Enter Year", 2018, 2022, 2022)

# Predict
encoded = brand_map[selected_med]
pred = model.predict(np.array([[encoded, year]]))[0]
st.success(f"Predicted Claims for {selected_med} in {year}: {int(pred)}")
