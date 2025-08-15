import pandas as pd
import numpy as np
import os


df = pd.read_csv("regression/global_cancer_patients_2015_2024.csv")

numeric_columns = [
    "Age", "Genetic_Risk", "Air_Pollution", "Alcohol_Use",
    "Smoking", "Obesity_Level", "Survival_Years"
]

def add_noise(df, noise_level=0.05, columns=None):
    df_noisy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        std = df_noisy[col].std()
        noise = np.random.normal(0, noise_level * std, size=len(df))
        df_noisy[col] += noise
    return df_noisy

output_dir = "regression/noisy_data"
os.makedirs(output_dir, exist_ok=True)

for noise_percent in [0.05, 0.10, 0.20]:
    noisy_df = add_noise(df, noise_level=noise_percent, columns=numeric_columns)
    filename = f"global_cancer_patients_noise_{int(noise_percent * 100)}.csv"
    noisy_df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Saved: {filename}")
