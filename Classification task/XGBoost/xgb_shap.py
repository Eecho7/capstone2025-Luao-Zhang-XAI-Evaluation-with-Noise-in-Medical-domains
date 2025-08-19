import os
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

if not hasattr(np, 'bool'):
    np.bool = np.bool_

noise_dir = "noisy_datasets_matrix"
output_model_dir = "saved_models_xgb"
output_report_dir = "classification_reports_xgb"
output_shap_dir = "shap_summary_plots_xgb"

os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(output_report_dir, exist_ok=True)
os.makedirs(output_shap_dir, exist_ok=True)


best_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1
}


test_df = pd.read_csv("Testing.csv")
X_test = test_df.drop(columns=["prognosis"])
y_test = test_df["prognosis"]


for file_name in sorted(os.listdir(noise_dir)):
    if not file_name.endswith(".csv"):
        continue

    print(f"\nProcessing: {file_name}")
    file_path = os.path.join(noise_dir, file_name)

    train_df = pd.read_csv(file_path)
    X_train = train_df.drop(columns=["prognosis"])
    y_train = train_df["prognosis"]
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        **best_params
    )
    model.fit(X_train, y_train_enc)

    model_path = os.path.join(output_model_dir, file_name.replace(".csv", ".pkl"))
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Accuracy: {acc:.4f}")

    report_dict = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(output_report_dir, file_name.replace(".csv", "_report.csv")))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False, max_display=10)

    leg = plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
        fontsize='small',
        title="Class", title_fontsize='x-small'
    )
    plt.gcf().tight_layout()

    shap_plot_path = os.path.join(output_shap_dir, file_name.replace(".csv", "_shap_top10.png"))
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Top 10 SHAP PLOT SAVED: {shap_plot_path}")
    
