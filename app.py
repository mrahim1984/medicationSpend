
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("medication_claim_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page title
st.title("💊 Medication Claim Predictor (2020)")

# Sidebar - Brand Selection
brand = st.selectbox("Select Brand Name", ["Panadol", "Tylenol", "GenericA", "GenericB", "Other"])

# Input fields
clm_2018 = st.number_input("Total Claims in 2018", min_value=0.0, step=1000.0)
clm_2019 = st.number_input("Total Claims in 2019", min_value=0.0, step=1000.0)
dosage_2018 = st.number_input("Dosage Units in 2018", min_value=0.0, step=1000.0)
dosage_2019 = st.number_input("Dosage Units in 2019", min_value=0.0, step=1000.0)

# Convert brand name to numeric
brand_map = {
    "Panadol": 1,
    "Tylenol": 2,
    "GenericA": 3,
    "GenericB": 4,
    "Other": 0
}
brand_encoded = brand_map[brand]

# Prediction
if st.button("Predict 2020 Claims"):
    input_data = np.array([[clm_2018, clm_2019, dosage_2018, dosage_2019, brand_encoded]])
    prediction = model.predict(input_data)[0]
    st.success(f"🧾 Predicted Tot_Clms_2020 for {brand}: {prediction:,.2f}")
