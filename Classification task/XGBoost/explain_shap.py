import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
 
model_paths = {
    "clean": "xgboost/xgb_model.pkl",  
    "noise_5": "xgboost/xgb_model_5.pkl",
    "noise_10": "xgboost/xgb_model_10.pkl",
    "noise_20": "xgboost/xgb_model_20.pkl"
}

X_paths = {
    "clean": "xgboost/X_train.csv",
    "noise_5": "xgboost/X_train_noisy_5.csv",
    "noise_10": "xgboost/X_train_noisy_10.csv",
    "noise_20": "xgboost/X_train_noisy_20.csv"
}

label_paths = {
    "clean": "xgboost/label.pkl",  
    "noise_5": "xgboost/label_5.pkl",
    "noise_10": "xgboost/label_10.pkl",
    "noise_20": "xgboost/label_20.pkl"
}
output_dir = "shap_summary_xgb"
os.makedirs(output_dir, exist_ok=True)

# Analyze each model in turn
for label in model_paths:
    print(f"\n Analyze model：{label}")
    model = joblib.load(model_paths[label])
    X = pd.read_csv(X_paths[label])

    X = X.select_dtypes(include=["int", "float"])

    # Initialize the interpreter
    explainer = shap.TreeExplainer(model, data=X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    # summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary ({label})")
    plt.savefig(f"{output_dir}/summary_{label}.png", bbox_inches='tight')
    plt.close()

    # bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Bar ({label})")
    plt.savefig(f"{output_dir}/bar_{label}.png", bbox_inches='tight')
    plt.close()

    print(f" {label} SHAP chart saved")