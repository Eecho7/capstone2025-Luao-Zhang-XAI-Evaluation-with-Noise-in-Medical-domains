import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("regression/global_cancer_patients_2015_2024.csv")

# Feature Engineering
X = df.drop(columns=[
    "Target_Severity_Score", "Patient_ID", "Cancer_Type",
    "Cancer_Stage", "Treatment_Cost_USD", "Country_Region"
])
X = pd.get_dummies(X).astype(float)
y = df["Target_Severity_Score"]

# Loading the model
model = joblib.load("regression/rf_regressor_model.pkl")

# Use KernelExplainer
explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50)) 
shap_values = explainer.shap_values(X.iloc[:100])  

# Output bar chart
plt.title("SHAP Bar (RF Regressor)")
shap.summary_plot(shap_values, X.iloc[:100], plot_type="bar", show=False)
plt.savefig("regression/shap_rf_bar.png", bbox_inches="tight")
plt.show()

# Output summary chart
plt.title("SHAP Summary (RF Regressor)")
shap.summary_plot(shap_values, X.iloc[:100], show=False)
plt.savefig("regression/shap_rf_summary.png", bbox_inches="tight")
plt.show()