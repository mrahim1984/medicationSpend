import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Define 30 real-sounding medication brand names
med_names = [
    "Panadol", "Aspirin", "Tylenol", "Zyrtec", "Claritin", "Advil", "Motrin", "Nurofen", "Aleve", "Benadryl",
    "Paracetamol", "Brufen", "Voltaren", "Sudafed", "Excedrin", "Codeine", "Ibuprofen", "Cataflam", "Loratadine",
    "Cetirizine", "Diclofenac", "Amoxicillin", "Augmentin", "Prednisone", "Omeprazole", "Esomeprazole", "Ranitidine",
    "Metformin", "Atorvastatin", "Simvastatin"
]

# Create dummy data
df = pd.DataFrame({
    'brand_encoded': np.random.randint(0, len(med_names), 150),
    'year': np.random.randint(2018, 2023, 150),
    'claims': np.random.randint(1000, 5000, 150)
})

X = df[['brand_encoded', 'year']]
y = df['claims']

# Train dummy Random Forest model
model = RandomForestRegressor()
model.fit(X, y)

# Create brand map
brand_map = {name: idx for idx, name in enumerate(med_names)}

st.title("💊 Medication Claim Predictor")

# Input medication and year
med = st.selectbox("Select Medication Name", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2018, max_value=2022, step=1)

# Predict
if st.button("Predict Next Year Claim"):
    encoded_med = brand_map[med]
    input_data = np.array([[encoded_med, year]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Claim for {med} in {year + 1}: {int(prediction[0])} claims")
