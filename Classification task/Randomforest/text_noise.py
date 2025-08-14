import pandas as pd
import joblib
from sklearn.metrics import classification_report

X_test = pd.read_csv("/Users/luao/Desktop/XAI PROJECT/rf/Testing.csv").drop(columns=["prognosis"])
y_test = pd.read_csv("/Users/luao/Desktop/XAI PROJECT/rf/Testing.csv")["prognosis"]

model_paths = {
    "clean (0%)": "rf_model.pkl",
    "noisy 5%": "rf_model_5.pkl",
    "noisy 10%": "rf_model_10.pkl",
    "noisy 20%": "rf_model_20.pkl"
}

for label, path in model_paths.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    print(f"\n Evaluation Report for {label}:")
    print(classification_report(y_test, y_pred))