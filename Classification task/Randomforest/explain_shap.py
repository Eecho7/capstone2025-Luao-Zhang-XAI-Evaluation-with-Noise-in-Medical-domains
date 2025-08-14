import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

model = joblib.load("rf_model.pkl")
X = pd.read_csv("X_train.csv")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap_vals_class1 = shap_values[1]

mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
feature_names = X.columns

top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
top_features = feature_names[top_indices]
top_values = mean_abs_shap[top_indices]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_features)), top_values, color="steelblue")
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value| (Feature Importance)")
ax.set_title("Top 15 Important Features (SHAP Bar Plot)")
plt.tight_layout()

plt.savefig("shap_bar_top15.png", dpi=300, bbox_inches="tight")
plt.close()

print(" SHAP Bar Chart（Top 15 Features）saved as shap_bar_top15.png")