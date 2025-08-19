import pandas as pd
import numpy as np
import os
import random
import itertools



def add_feature_noise(df, patient_percent, feature_percent):
    df_noisy = df.copy()
    feature_cols = df_noisy.columns.drop("prognosis")
    if patient_percent == 0 or feature_percent == 0:
        print("    - Feature Noise: Level 0, no features flipped.")
        return df_noisy
    num_patients = int(len(df_noisy) * (patient_percent / 100))
    num_features = int(len(feature_cols) * (feature_percent / 100))
    selected_patients_indices = df_noisy.sample(n=num_patients, random_state=42).index
    total_flips = 0
    for idx in selected_patients_indices:
        selected_features = random.sample(list(feature_cols), num_features)
        for feat in selected_features:
            if df_noisy.at[idx, feat] in [0, 1]:
                df_noisy.at[idx, feat] = 1 - df_noisy.at[idx, feat]
                total_flips += 1
    print(f"    - Feature Noise: Flipped {total_flips} feature values.")
    return df_noisy


def add_label_noise(df, label_noise_percent):
    df_noisy = df.copy()
    prognosis_col = 'prognosis'
    if label_noise_percent == 0:
        print("    - Label Noise: Level 0, no labels flipped.")
        return df_noisy
    unique_labels = df_noisy[prognosis_col].unique()
    n_samples_to_noise = int((label_noise_percent / 100) * len(df_noisy))
    noise_indices = df_noisy.sample(n=n_samples_to_noise, random_state=43).index
    flipped_count = 0
    for idx in noise_indices:
        original_label = df_noisy.at[idx, prognosis_col]
        potential_new_labels = [label for label in unique_labels if label != original_label]
        if potential_new_labels:
            new_label = random.choice(potential_new_labels)
            df_noisy.at[idx, prognosis_col] = new_label
            flipped_count += 1
    print(f"    - Label Noise: Flipped {flipped_count} labels.")
    return df_noisy


# Main


base_train_path = "train_cleaned.csv"
output_dir = "noisy_datasets_matrix"  
os.makedirs(output_dir, exist_ok=True)

try:
    df_base = pd.read_csv(base_train_path)
except FileNotFoundError:
    print(f"ERROR: '{base_train_path}'.")
    exit()


p_percents = [0, 25, 50, 75]

f_percents = [0, 25, 50, 75]

l_percents = [0, 10, 20, 30]


total_files = len(p_percents) * len(f_percents) * len(l_percents)


file_counter = 0
for p in p_percents:
    for f in f_percents:
        for l in l_percents:
            file_counter += 1
            if p == 0 and f == 0 and l == 0:
                if file_counter > 1: 
                    continue

            print(f"\n({file_counter}/{total_files}) : P={p}%, F={f}%, L={l}%")

    
            df_feature_noisy = add_feature_noise(df_base, p, f)
           
            df_combined_noisy = add_label_noise(df_feature_noisy, l)

            file_name = f"matrix_p{p}_f{f}_l{l}.csv"
            file_path = os.path.join(output_dir, file_name)
            df_combined_noisy.to_csv(file_path, index=False)
            print(f"SAVED: {file_name}")

print(f"\nSuccessfully generated {total_files} matrices.")
print(f"Saved in '{output_dir}' folder.")