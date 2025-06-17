
import streamlit as st
import pickle
import numpy as np

# Load model
with open("medication_claim_model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Expanded dummy brand names
brand_names = {
    "Aspirin": 0,
    "Paracetamol": 1,
    "Ibuprofen": 2,
    "Amoxicillin": 3,
    "Atorvastatin": 4,
    "Metformin": 5,
    "Losartan": 6,
    "Omeprazole": 7,
    "Simvastatin": 8,
    "Levothyroxine": 9,
    "Azithromycin": 10,
    "Hydrochlorothiazide": 11,
    "Gabapentin": 12,
    "Lisinopril": 13,
    "Metoprolol": 14
}

# Streamlit UI
st.title("💊 Predict Medication Claim Volume (Random Forest)")

med_name = st.selectbox("Select Medication Name", list(brand_names.keys()))
year = st.number_input("Enter Year", min_value=2018, max_value=2022, step=1)

if st.button("Predict Claims"):
    brand_encoded = brand_names[med_name]
    input_data = np.array([[brand_encoded, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Claims for {med_name} in {year}: {prediction:,.0f}")
