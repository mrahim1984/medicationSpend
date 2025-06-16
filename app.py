
import streamlit as st
import pickle
import numpy as np

# Load model and brand map
with open("medication_claim_model.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

# Streamlit UI
st.title("💊 Predict Medication Claim Volume (Random Forest)")

# Inputs
med_name = st.selectbox("Choose Medication", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1)

# Predict button
if st.button("Predict Next Year Claims"):
    if med_name in brand_map:
        encoded = brand_map[med_name]
        features = np.array([[encoded, year]])
        prediction = model.predict(features)[0]
        st.success(f"📈 Predicted claims for {med_name} in {year + 1}: {int(prediction):,}")
    else:
        st.error("Medication not found in the model.")
