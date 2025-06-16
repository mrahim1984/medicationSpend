
import streamlit as st
import pickle
import numpy as np

# Load model
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# UI
st.title("💊 Medication Claim Predictor")
st.markdown("Predict next year's claim volume for a specific medication.")

med_name = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year (≤ 2022)", min_value=2018, max_value=2022, value=2022, step=1)

if st.button("Predict Next Year Claim"):
    if med_name in brand_map:
        brand_encoded = brand_map[med_name]
        input_data = np.array([[brand_encoded, year]])
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted Claims for {med_name} in {year + 1}: {prediction:,.0f}")
    else:
        st.error("❌ Medication not found.")
