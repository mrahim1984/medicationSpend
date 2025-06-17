import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Create dummy data
df = pd.DataFrame({
    'brand_encoded': np.random.randint(0, 15, 100),
    'year': np.random.randint(2018, 2023, 100),
    'claims': np.random.randint(1000, 5000, 100)
})

X = df[['brand_encoded', 'year']]
y = df['claims']

# Train dummy Random Forest model
model = RandomForestRegressor()
model.fit(X, y)

# Generate dummy brand map
brand_map = {f"Med_{i}": i for i in range(15)}

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
