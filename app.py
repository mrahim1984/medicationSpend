
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create dummy data
df = pd.DataFrame({
    'brand_encoded': np.random.randint(0, 5, 100),
    'year': np.random.randint(2018, 2022, 100),
    'claims': np.random.randint(1000, 5000, 100)
})

X = df[['brand_encoded', 'year']]
y = df['claims']

# Train dummy XGBoost model
model = xgb.XGBRegressor()
model.fit(X, y)

# Save model
with open("medication_claim_model.pkl", "wb") as f:
    pickle.dump(model, f)
