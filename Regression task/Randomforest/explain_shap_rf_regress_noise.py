import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

#Preset Path Templates
noise_levels = [0, 5, 10, 20]
data_path_template = "regression/noisy_data/global_cancer_patients_noise_{}.csv"
model_path_template = "regression/rf_noise_models/rf_model_noise_{}.pkl"
output_dir = "regression/shap_plots_regression"

#Output
os.makedirs(output_dir, exist_ok=True)

for noise in noise_levels:
    print(f"\n Processing noise level: {noise}%")

    # Path Definition
    data_path = data_path_template.format(noise)
    model_path = model_path_template.format(noise)

    # Load data
    df = pd.read_csv(data_path)

    # Feature engineering
    X = df.drop(columns=[
        "Target_Severity_Score", "Patient_ID", "Cancer_Type",
        "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"
    ])
    X = pd.get_dummies(X).astype(float)

    # Only take the first 100 rows to avoid crashes
    X_subset = X.iloc[:100]

    # Load the model
    model = joblib.load(model_path)

    # SHAP analysis (using KernelExplainer for best compatibility)
    explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
    shap_values = explainer.shap_values(X_subset)

    # SHAP bar chart
    plt.title(f"SHAP Bar (Noise {noise}%)")
    shap.summary_plot(shap_values, X_subset, plot_type="bar", show=False)
    plt.savefig(f"{output_dir}/shap_rf_bar_{noise}.png", bbox_inches="tight")
    plt.close()

    # SHAP Summary chart
    plt.title(f"SHAP Summary (Noise {noise}%)")
    shap.summary_plot(shap_values, X_subset, show=False)
    plt.savefig(f"{output_dir}/shap_rf_summary_{noise}.png", bbox_inches="tight")
    plt.close()

    print(f" SHAP graph saved: noise={noise}%")