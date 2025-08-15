import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import numpy as np

noise_levels = [0, 5, 10, 20]
data_path_template = "regression/noisy_data/global_cancer_patients_noise_{}.csv"
model_output_dir = "regression/xgb_noise_models"
os.makedirs(model_output_dir, exist_ok=True)

for noise in noise_levels:
    print(f"\n Training XGBoost Regressor - Noise Level: {noise}%")

    data_path = data_path_template.format(noise)
    df = pd.read_csv(data_path)

    # Feature enginering
    X = df.drop(columns=[
        "Target_Severity_Score", "Patient_ID", "Cancer_Type",
        "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"
    ])
    X = pd.get_dummies(X).astype(float)
    y = df["Target_Severity_Score"]

    # Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Performance Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f" RMSE (noise={noise}%): {rmse:.4f}")

    # Save modeol
    model_path = f"{model_output_dir}/xgb_model_noise_{noise}.pkl"
    joblib.dump(model, model_path)
    print(f" Model saved in: {model_path}")

    # Save feature
    feature_path = f"{model_output_dir}/feature_noise_{noise}.pkl"
    joblib.dump(X.columns.tolist(), feature_path)
    print(f" Feature list saved in: {feature_path}")

    print(f" Train finished - Noise level: {noise}%")
