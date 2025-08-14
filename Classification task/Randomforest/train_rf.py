import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("X_train_noisy.csv")
y = pd.read_csv("y_train.csv")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "rf_model.pkl")
print(" MOdel saved as rf_model.pkl")