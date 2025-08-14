# train_xgb_noise.py

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def train_and_save_model(X_path, y_path, model_path, label_path):
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path, header=None).squeeze()

    
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    clf.fit(X, y_enc)

    
    joblib.dump(clf, model_path)
    joblib.dump(le, label_path)
    print(f" Model saved to {model_path}")
    print(f" Tag encoder saved to {label_path}")


train_and_save_model("xgboost/X_train.csv", "xgboost/y_train.csv", "xgboost/xgb_model.pkl", "xgboost/label.pkl")
train_and_save_model("xgboost/X_train_noisy_5.csv", "xgboost/y_train.csv", "xgboost/xgb_model_5.pkl", "xgboost/label_5.pkl")
train_and_save_model("xgboost/X_train_noisy_10.csv", "xgboost/y_train.csv", "xgboost/xgb_model_10.pkl", "xgboost/label_10.pkl")
train_and_save_model("xgboost/X_train_noisy_20.csv", "xgboost/y_train.csv", "xgboost/xgb_model_20.pkl", "xgboost/label_20.pkl")
