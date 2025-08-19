import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re


noise_data_dir = "noisy_datasets_matrix"
test_data_path = "Testing.csv"


results = []
all_files = [f for f in os.listdir(noise_data_dir) if f.endswith('.csv')]


def sort_key_func(file_name):
    parts = re.findall(r'\d+', file_name)
    p, f, l = map(int, parts)
    return (l, p, f)


all_files.sort(key=sort_key_func)

print(f"--- Start batch evaluation ---")

test_df = pd.read_csv(test_data_path)
X_test = test_df.drop(columns=["prognosis"])
y_test = test_df["prognosis"]

for train_file in all_files:
    print(f"\nProcessing: {train_file}...")

    train_df = pd.read_csv(os.path.join(noise_data_dir, train_file))
    X_train = train_df.drop(columns=["prognosis"])
    y_train = train_df["prognosis"]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train_enc)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_enc, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)

    results.append({
        "Dataset": train_file,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


if results:
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values(by="F1 Score", ascending=False)

    print("\n\n" + "=" * 60)
    print("Model Performance Evaluation Report")
    print("=" * 60)
    print(results_df_sorted.to_string(index=False))
    print("=" * 60)