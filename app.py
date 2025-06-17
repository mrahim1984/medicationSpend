
import streamlit as st
import pickle
import numpy as np

# Load model and brand map
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Reverse brand map for dropdown
reverse_map = {v: k for k, v in brand_map.items()}

# Streamlit UI
st.title("💊 Medication Claim Predictor")
med_name = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, value=2022)

if st.button("Predict Next Year Claim"):
    brand_encoded = brand_map[med_name]
    input_data = np.array([[brand_encoded, year]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Claims for {med_name} in {year}: {int(prediction[0])}")
