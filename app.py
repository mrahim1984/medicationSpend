
import streamlit as st
import pickle
import numpy as np

# Load model and brand map
with open("medication_claim_model_rf.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# App title
st.title("💊 Medication Claim Predictor")

# Brand input
med_names = list(brand_map.keys())
selected_brand = st.selectbox("Select Medication Name", med_names)

# Year input
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1)

# Prediction button
if st.button("Predict Next Year Claim"):
    brand_encoded = brand_map[selected_brand]
    input_data = np.array([[brand_encoded, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Claims for {selected_brand} in {year + 1}: {int(prediction)}")
