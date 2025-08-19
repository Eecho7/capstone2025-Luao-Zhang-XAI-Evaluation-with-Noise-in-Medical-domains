import pandas as pd
import os
import re
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


noise_data_dir = "noisy_datasets_full"
test_data_path = os.path.join("split_regression_data_combined", "test_set.csv")
main_output_dir = "analysis_results_xgboost" 
TARGET_VARIABLE = 'Target_Severity_Score'

os.makedirs(os.path.join(main_output_dir, "summary_plots"), exist_ok=True)
os.makedirs(os.path.join(main_output_dir, "bar_plots"), exist_ok=True)
os.makedirs(os.path.join(main_output_dir, "dependence_plots"), exist_ok=True)
os.makedirs(os.path.join(main_output_dir, "waterfall_plots"), exist_ok=True)
os.makedirs(os.path.join(main_output_dir, "force_plot_html"), exist_ok=True)



results = []
all_files = [f for f in os.listdir(noise_data_dir) if f.endswith('.csv')]


def sort_key_func(file_name):
    parts = re.findall(r'\d+', file_name)
    n, c, l = map(int, parts)
    return (l, c, n)


all_files.sort(key=sort_key_func)

print(f"Batch evaluation and generating 5 types of SHAP plots")

test_df = pd.read_csv(test_data_path)
y_test = test_df[TARGET_VARIABLE]
X_test = test_df.drop(columns=[TARGET_VARIABLE])
idx_for_waterfall = y_test.idxmax()
ROWS_TO_USE = 500

for train_file in all_files:
    print(f"\nProcessing: {train_file}...")

    train_df = pd.read_csv(os.path.join(noise_data_dir, train_file))

    if ROWS_TO_USE > 0:
        print(f"Using first {ROWS_TO_USE} rows for training to speed up.")
        train_df = train_df.head(ROWS_TO_USE)

    y_train = train_df[TARGET_VARIABLE]
    X_train = train_df.drop(columns=[TARGET_VARIABLE])
    X_train['is_train'] = 1
    X_test_copy = X_test.copy()
    X_test_copy['is_train'] = 0
    combined_X = pd.concat([X_train, X_test_copy], ignore_index=True)
    combined_X_encoded = pd.get_dummies(combined_X)
    X_train_processed = combined_X_encoded[combined_X_encoded['is_train'] == 1].drop('is_train', axis=1)
    X_test_processed = combined_X_encoded[combined_X_encoded['is_train'] == 0].drop('is_train', axis=1)

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    r2, mse, mae = r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    results.append({"数据集": train_file, "R2 Score": r2, "MSE": mse, "MAE": mae})
    print(f"  -> R2 Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # SHAP

    print(f"Performing full SHAP analysis for {train_file}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_processed)

    # Summary Plot
    shap.summary_plot(shap_values, X_test_processed, show=False)
    plt.title(f'Summary Plot (XGBoost) for {train_file}');
    plt.tight_layout()
    plt.savefig(os.path.join(main_output_dir, "summary_plots", f"summary_{train_file.replace('.csv', '.png')}"));
    plt.close()
    print(f"    - 1/5 Summary plot saved.")

    # Bar Plot
    shap.summary_plot(shap_values, X_test_processed, plot_type="bar", show=False)
    plt.title(f'Bar Plot (XGBoost) for {train_file}');
    plt.tight_layout()
    plt.savefig(os.path.join(main_output_dir, "bar_plots", f"bar_{train_file.replace('.csv', '.png')}"));
    plt.close()
    print(f"    - 2/5 Bar plot saved.")

    # Dependence Plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_3_indices = np.argsort(mean_abs_shap)[-3:]
    top_3_features = X_test_processed.columns[top_3_indices]
    for feature in top_3_features:
        shap.dependence_plot(feature, shap_values, X_test_processed, interaction_index="auto", show=False)
        plt.title(f'Dependence Plot for "{feature}"\n(Model: {train_file})');
        plt.tight_layout()
        plt.savefig(os.path.join(main_output_dir, "dependence_plots",
                                 f"dependence_{feature}_{train_file.replace('.csv', '.png')}"));
        plt.close()

    # Waterfall Plot
    shap.waterfall_plot(shap.Explanation(values=shap_values[idx_for_waterfall], base_values=explainer.expected_value,
                                         data=X_test_processed.iloc[idx_for_waterfall],
                                         feature_names=X_test_processed.columns), show=False)
    plt.title(f'Waterfall Plot for Most Severe Case\n(Model: {train_file})');
    plt.tight_layout()
    plt.savefig(os.path.join(main_output_dir, "waterfall_plots", f"waterfall_{train_file.replace('.csv', '.png')}"));
    plt.close()


if results:
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    print("\n\n" + "=" * 60)
    print("Final evaluation report of XGBoost regression model")
    print("=" * 60)
    results_df['R2 Score'] = results_df['R2 Score'].map('{:.4f}'.format)
    results_df['MSE'] = results_df['MSE'].map('{:.2f}'.format)
    results_df['MAE'] = results_df['MAE'].map('{:.2f}'.format)
    print(results_df.to_string(index=False))
    print("=" * 60)
