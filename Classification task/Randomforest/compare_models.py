import joblib
import pandas as pd
from sklearn.metrics import classification_report

# Loading the test set
X_test = pd.read_csv("rf/Testing.csv").drop(columns=["prognosis"])
y_test = pd.read_csv("rf/Testing.csv")["prognosis"]

# Loading two model
model_clean = joblib.load("rf_model.pkl")       
model_noisy = joblib.load("rf_model_5.pkl")     

# Get result
y_pred_clean = model_clean.predict(X_test)
y_pred_noisy = model_noisy.predict(X_test)

# compare two model
comparison_df = pd.DataFrame({
    "True Label": y_test,
    "Pred_Clean": y_pred_clean,
    "Pred_Noisy": y_pred_noisy
})

# Find differnence in two model
diff = comparison_df[
    (comparison_df["Pred_Clean"] != comparison_df["True Label"]) &
    (comparison_df["Pred_Noisy"] == comparison_df["True Label"])
]

wrong_by_noise = comparison_df[
    (comparison_df["Pred_Clean"] == comparison_df["True Label"]) &
    (comparison_df["Pred_Noisy"] != comparison_df["True Label"])
]

# Save and print result
diff.to_csv("fixed_by_noise.csv", index=False)
print(" The forecast revision sample is as follows：")
print(diff)

# Print evaluation report for each model
print("\n Evaluation Report - Clean Model:")
print(classification_report(y_test, y_pred_clean))

print("\n Evaluation Report - Noisy Model (5%):")
print(classification_report(y_test, y_pred_noisy))