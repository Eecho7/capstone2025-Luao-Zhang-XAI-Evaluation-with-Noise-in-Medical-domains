
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(X_path, y_path, model_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path, header=None).squeeze()  
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, model_path)
    print(f" Model saved as {model_path}")


train_and_save_model("X_train.csv", "y_train.csv", "rf_model.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p10_f10.csv", "y_train.csv", "rf_model_p10_f10.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p10_f20.csv", "y_train.csv", "rf_model_p10_f20.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p10_f30.csv", "y_train.csv", "rf_model_p10_f30.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p10_f40.csv", "y_train.csv", "rf_model_p10_f40.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p10_f50.csv", "y_train.csv", "rf_model_p10_f50.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p20_f10.csv", "y_train.csv", "rf_model_p20_f10.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p20_f20.csv", "y_train.csv", "rf_model_p20_f20.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p20_f30.csv", "y_train.csv", "rf_model_p20_f30.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p20_f40.csv", "y_train.csv", "rf_model_p20_f40.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p20_f50.csv", "y_train.csv", "rf_model_p20_f50.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p30_f10.csv", "y_train.csv", "rf_model_p30_f10.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p30_f20.csv", "y_train.csv", "rf_model_p30_f20.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p30_f30.csv", "y_train.csv", "rf_model_p30_f30.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p30_f40.csv", "y_train.csv", "rf_model_p30_f40.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p30_f50.csv", "y_train.csv", "rf_model_p30_f50.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p40_f10.csv", "y_train.csv", "rf_model_p40_f10.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p40_f20.csv", "y_train.csv", "rf_model_p40_f20.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p40_f30.csv", "y_train.csv", "rf_model_p40_f30.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p40_f40.csv", "y_train.csv", "rf_model_p40_f40.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p40_f50.csv", "y_train.csv", "rf_model_p40_f50.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p50_f10.csv", "y_train.csv", "rf_model_p50_f10.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p50_f20.csv", "y_train.csv", "rf_model_p50_f20.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p50_f30.csv", "y_train.csv", "rf_model_p50_f30.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p50_f40.csv", "y_train.csv", "rf_model_p50_f40.pkl")
train_and_save_model("noisy_datasets_grid/train_noise_p50_f50.csv", "y_train.csv", "rf_model_p50_f50.pkl")

