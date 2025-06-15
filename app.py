
import streamlit as st
import pickle
import numpy as np

with open("medication_claim_model.pkl", "rb") as f:
    model = pickle.load(f)

import pickle

with open("medication_claim_model.pkl", "wb") as f:
    pickle.dump((model, brand_map), f)

# App UI
st.title("💊 Predict Medication Claim Volume for Next Year")

# User inputs
med_name = st.text_input("Enter Medication Name")
year = st.number_input("Enter Current Year", min_value=2000, max_value=2100, step=1)

if st.button("Predict Next Year Claim"):
    if med_name in brand_map:
        brand_encoded = brand_map[med_name]
        input_data = np.array([[brand_encoded, year]])
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted claims for {med_name} in {year + 1}: {prediction:,.2f}")
    else:
        st.error(f"❌ Medication '{med_name}' not found in model. Try another name.")
