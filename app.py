import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy model training (just for testing)
X_train = np.array([
    [1, 2018],
    [2, 2019],
    [3, 2020],
    [4, 2021],
])
y_train = np.array([100, 200, 300, 400])

model = LinearRegression()
model.fit(X_train, y_train)

# Manual brand map
brand_map = {
    "Panadol": 1,
    "Tylenol": 2,
    "GenericA": 3,
    "GenericB": 4,
    "Other": 0
}

# UI
st.title("💊 Medication Claim Predictor (Dummy Model)")

brand = st.selectbox("Select Brand", list(brand_map.keys()))
year = st.number_input("Enter Current Year", min_value=2010, max_value=2100, step=1)

if st.button("Predict"):
    brand_encoded = brand_map[brand]
    input_data = np.array([[brand_encoded, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted claims for {brand} in {year + 1}: {prediction:,.2f}")

