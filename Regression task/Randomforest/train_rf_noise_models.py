import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

input_path = "regression/global_cancer_patients_2015_2024.csv"
model_output_dir = "regression/rf_noise_models"
os.makedirs(model_output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_path)


drop_cols = ["Target_Severity_Score", "Patient_ID", "Cancer_Type", "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"]
X_base = df.drop(columns=drop_cols)
X_base = pd.get_dummies(X_base).astype(float)
y = df["Target_Severity_Score"]

# Split training and test sets）
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

# Noise ratio to be tested
noise_levels = [0.0, 0.05, 0.10, 0.20]

# Save result
results = {}

for level in noise_levels:
    
    X_train = X_train_base.copy()
    X_test = X_test_base.copy()

    if level > 0:
        stds = X_train.std()
        noise_train = np.random.normal(0, stds * level, size=X_train.shape)
        noise_test = np.random.normal(0, stds * level, size=X_test.shape)
        X_train += noise_train
        X_test += noise_test

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[f"noise_{int(level * 100)}%"] = rmse
    print(f" Noise {int(level * 100)}% - RMSE: {rmse:.4f}")

    # Save model
    model_path = os.path.join(model_output_dir, f"rf_model_noise_{int(level * 100)}.pkl")
    joblib.dump(model, model_path)

# Performance Summary (RMSE)
print("\n Performance Summary (RMSE):")
for k, v in results.items():
    print(f"{k:<10}: {v:.4f}")