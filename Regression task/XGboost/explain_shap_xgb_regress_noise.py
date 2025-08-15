import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# 配置路径
noise_levels = [0, 5, 10, 20]
data_path_template = "regression/noisy_data/global_cancer_patients_noise_{}.csv"
model_path_template = "regression/xgb_noise_models/xgb_model_noise_{}.pkl"
feature_path_template = "regression/xgb_noise_models/feature_noise_{}.pkl"
output_dir = "regression/shap_plots_xgb"
os.makedirs(output_dir, exist_ok=True)

TARGET_COL = "Target_Severity_Score"
DROP_COLS = [
    "Target_Severity_Score", "Patient_ID", "Cancer_Type",
    "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"
]

# Main
for noise in noise_levels:
    print(f"\n SHAP analying – Noise Level: {noise}%")

    data_path = data_path_template.format(noise)
    model_path = model_path_template.format(noise)
    feature_path = feature_path_template.format(noise)

    # Load data
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)

    # Feature engineering
    X = df.drop(columns=DROP_COLS)
    X = pd.get_dummies(X).astype(float)

    # Complete missing columns (to prevent SHAP errors)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]  

# SHAP analysis (first 100 samples)
    background = shap.sample(X, 50)
    explain_X = X.iloc[:100]

    explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    shap_values = explainer.shap_values(explain_X)

    background = shap.sample(X, 50)
    explain_X = X.iloc[:100]

    explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
    shap_values = explainer.shap_values(explain_X)

    bar_path = os.path.join(output_dir, f"xgb_shap_bar_{noise}.png")
    summary_path = os.path.join(output_dir, f"xgb_shap_summary_{noise}.png")

# Bar chart
    plt.title(f"SHAP Bar (XGBoost, Noise {noise}%)")
    shap.summary_plot(shap_values, explain_X, plot_type="bar", show=False)
    plt.savefig(bar_path, bbox_inches="tight")
    plt.clf()

# Summary chart
    plt.title(f"SHAP Summary (XGBoost, Noise {noise}%)")
    shap.summary_plot(shap_values, explain_X, show=False)
    plt.savefig(summary_path, bbox_inches="tight")
    plt.clf()

    print(" All noise model SHAP analysis completed")