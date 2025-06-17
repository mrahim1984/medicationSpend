import streamlit as st
import pickle
import numpy as np

# Load model and brand map
with open("medication_claim_model_rf.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Reverse the brand map for display
brand_map_reverse = {v: k for k, v in brand_map.items()}

st.title("💊 Medication Claim Predictor")

# Select brand name
brand_name = st.selectbox("Select Medication Name", list(brand_map.keys()))
current_year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1, value=2022)

if st.button("Predict Next Year Claim"):
    brand_encoded = brand_map[brand_name]
    input_data = np.array([[brand_encoded, current_year]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Claims for {brand_name} in {current_year + 1}: {int(prediction)}")
