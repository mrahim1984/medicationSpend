
import streamlit as st
import pickle
import numpy as np

# Load model and brand_map
with open("medication_claim_model_rf.pkl", "rb") as f:
    model, brand_map = pickle.load(f)

st.title("💊 Medication Claim Predictor")

# Input medication and year
med = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1)

if st.button("Predict Next Year Claim"):
    encoded_brand = brand_map[med]
    X_pred = np.array([[encoded_brand, year]])
    prediction = model.predict(X_pred)
    st.success("📈 Predicted Claim Volume for Next Year: {}".format(int(prediction[0])))
