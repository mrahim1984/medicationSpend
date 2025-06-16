
import streamlit as st
import pickle
import numpy as np

# Load model and brand map
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Streamlit UI
st.title("💊 Medication Claim Predictor")
st.write("Predict next-year claim volume based on medication and year.")

# User inputs
medications = list(brand_map.keys())
selected_med = st.selectbox("Select Medication Name", medications)
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, value=2022, step=1)

if st.button("Predict Next Year Claim"):
    brand_encoded = brand_map[selected_med]
    input_features = np.array([[brand_encoded, year]])
    prediction = model.predict(input_features)
    st.success(f"Predicted Claim Count for {selected_med} in {year + 1}: {int(prediction[0])}")
