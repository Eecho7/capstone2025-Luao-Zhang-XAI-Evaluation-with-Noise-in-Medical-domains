import os
import pandas as pd
import shap
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


noise_dir = "noisy_datasets_matrix"
output_model_dir = "saved_models"
output_shap_dir = "shap_summary_plots"

os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(output_shap_dir, exist_ok=True)


test_df = pd.read_csv("Testing.csv")
X_test = test_df.drop(columns=["prognosis"])
y_test = test_df["prognosis"]


for file_name in sorted(os.listdir(noise_dir)):
    if not file_name.endswith(".csv"):
        continue

    print(f"Processing: {file_name}")
    file_path = os.path.join(noise_dir, file_name)


    train_df = pd.read_csv(file_path)
    X_train = train_df.drop(columns=["prognosis"])
    y_train = train_df["prognosis"]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train_enc)

    model_name = file_name.replace(".csv", ".pkl")
    joblib.dump(model, os.path.join(output_model_dir, model_name))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 6))

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_train, show=False, max_display=10)
    else:
        shap.summary_plot(shap_values, X_train, show=False, max_display=10)

    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    shap_plot_path = os.path.join(output_shap_dir, file_name.replace(".csv", "_shap_top10.png"))
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Top 10 SHAP PLOT SAVED: {shap_plot_path}")