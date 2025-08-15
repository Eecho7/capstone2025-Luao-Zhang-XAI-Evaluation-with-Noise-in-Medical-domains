import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv("regression/global_cancer_patients_2015_2024.csv")

# Feature engeniring
X = df.drop(columns=[
    "Target_Severity_Score", "Patient_ID", "Cancer_Type",
    "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"
])
X = pd.get_dummies(X).astype(float)
y = df["Target_Severity_Score"]

# Save feature name
joblib.dump(X.columns.tolist(), "regression/feature_columns_rf.pkl")

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "regression/rf_regressor_model.pkl")

# Performance Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(" RMSE:", rmse)