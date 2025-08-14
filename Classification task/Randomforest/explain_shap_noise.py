import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

configs = [
    ("rf_model_5.pkl", "X_train_noisy_5.csv", "shap_bar_top15_5.png"),
    ("rf_model_10.pkl", "X_train_noisy_10.csv", "shap_bar_top15_10.png"),
    ("rf_model_20.pkl", "X_train_noisy_20.csv", "shap_bar_top15_20.png"),
]

for model_path, data_path, output_path in configs:
    print(f" Procing {data_path} ...")
    model = joblib.load(model_path)
    X = pd.read_csv(data_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values[1], X, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()
    print(f" Saved {output_path}")