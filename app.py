import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("medication_claim_model.pkl", "rb") as f:
    model = pickle.load(f)

# Mapping
brand_map = {
    "Panadol": 1,
    "Tylenol": 2,
    "GenericA": 3,
    "GenericB": 4,
    "Other": 0
}

# App title
st.title("💊 Medication Claim Predictor")

# Inputs
brand = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2000, max_value=2100, step=1)

# Predict
if st.button("Predict Next Year Claim"):
    brand_encoded = brand_map[brand]
    input_data = np.array([[brand_encoded, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted claims for {brand} in {year + 1}: {prediction:,.2f}")

