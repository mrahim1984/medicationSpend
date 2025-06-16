
import streamlit as st
import pickle
import numpy as np

# Load model
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Reverse map for dropdown
reverse_map = {v: k for k, v in brand_map.items()}

# UI
st.title("💊 Medication Claim Predictor")

brand = st.selectbox("Select Medication Name", options=list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, value=2020)

if st.button("Predict Next Year Claim"):
    encoded_brand = brand_map[brand]
    next_year = year + 1
    prediction = model.predict(np.array([[encoded_brand, next_year]]))
    st.success(f"Predicted Claims for {brand} in {next_year}: {int(prediction[0])}")
