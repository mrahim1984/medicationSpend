
import streamlit as st
import numpy as np
import pickle

# Load model and brand_map
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Title
st.title("💊 Medication Claim Predictor")

# Medication options
medications = list(brand_map.keys())
selected_med = st.selectbox("Select Medication Name", sorted(medications))

# Input year
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, value=2022, step=1)

# Predict button
if st.button("Predict Next Year Claim"):
    if selected_med not in brand_map:
        st.error("Medication not found in model.")
    else:
        input_data = np.array([[brand_map[selected_med], year]])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Insurance Claims for {selected_med} in {year}: {int(prediction)}")
