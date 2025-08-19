import pandas as pd
import numpy as np
import os
import random


def add_numeric_noise(df, numeric_feature_level, label_level, target_column):
    df_noisy = df.copy()

    numeric_feature_cols = df_noisy.select_dtypes(include=np.number).columns
    cols_to_exclude = [target_column, 'Patient_ID']
    numeric_feature_cols = [col for col in numeric_feature_cols if col not in cols_to_exclude]

    if numeric_feature_level > 0:
        for col in numeric_feature_cols:
            std = df_noisy[col].std()
            if std > 0:
                noise = np.random.normal(0, std * numeric_feature_level, len(df_noisy))
                df_noisy[col] = df_noisy[col] + noise
        print(f"    - Numeric Feature Noise: Applied level {numeric_feature_level:.2f}.")

    if label_level > 0:
        std = df_noisy[target_column].std()
        if std > 0:
            noise = np.random.normal(0, std * label_level, len(df_noisy))
            df_noisy[target_column] = df_noisy[target_column] + noise
            print(f"    - Label Noise: Applied level {label_level:.2f} to target '{target_column}'.")

    return df_noisy


def add_categorical_noise(df, categorical_noise_level):
    df_noisy = df.copy()

    categorical_cols = df_noisy.select_dtypes(exclude=np.number).columns

    if categorical_noise_level > 0 and not categorical_cols.empty:
        n_samples_to_noise = int(len(df_noisy) * (categorical_noise_level / 100))
        noise_indices = df_noisy.sample(n=n_samples_to_noise, random_state=44).index

        flipped_count = 0
        for idx in noise_indices:
            col_to_flip = random.choice(categorical_cols)

            original_value = df_noisy.at[idx, col_to_flip]
            unique_values = df_noisy[col_to_flip].unique()
            potential_new_values = [val for val in unique_values if val != original_value]

            if potential_new_values:
                new_value = random.choice(potential_new_values)
                df_noisy.at[idx, col_to_flip] = new_value
                flipped_count += 1

        print(f"Categorical Noise: Flipped {flipped_count} categorical feature values.")

    return df_noisy


#Main


TARGET_VARIABLE = 'Target_Severity_Score'

numeric_feature_levels = [0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%
categorical_feature_levels = [0, 10, 20, 30]  # 0%, 10%, 20%
label_levels = [0, 0.1, 0.2, 0.3]  # 0%, 10%, 20%


base_train_path = os.path.join("split_regression_data_combined", "train_set.csv")
output_dir = "noisy_datasets_full"
os.makedirs(output_dir, exist_ok=True)


total_files = len(numeric_feature_levels) * len(categorical_feature_levels) * len(label_levels)

file_counter = 0
for n_level in numeric_feature_levels:
    for c_level in categorical_feature_levels:
        for l_level in label_levels:
            file_counter += 1

            if n_level == 0 and c_level == 0 and l_level == 0 and file_counter > 1:
                continue

            print(
                f"\n({file_counter}/{total_files}) Generating configuration: Numeric_F={n_level:.2f}, Categorical_F={c_level}%, Label_L={l_level:.2f}")

            df_cat_noisy = add_categorical_noise(df_base, c_level)

            df_full_noisy = add_numeric_noise(df_cat_noisy, n_level, l_level, target_column=TARGET_VARIABLE)

            file_name = f"full_noise_n{int(n_level * 100)}_c{c_level}_l{int(l_level * 100)}.csv"
            file_path = os.path.join(output_dir, file_name)
            df_full_noisy.to_csv(file_path, index=False)
            print(f"Saved: {file_name}")

print(f"\nAll {total_files} complete noise datasets generated successfully")
print(f"Files saved in '{output_dir}' folder.")
