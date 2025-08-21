# capstone2025-Luao-Zhang-XAI-Evaluation-with-Noise-in-Medical-domains
This project studies how noise affects both predictive performance and interpretability of machine learning models in medical settings. We evaluate Random Forest and XGBoost under multi-dimensional noise and analyze explanations using SHAP.

## Project Structure
```
Classification task/                
│
├── Randomforest/
│   ├── rf_train.py         # Train Random Forest classification model
│   └── RF_SHAP.py          # SHAP analyze with Random Forest classification model
│
├── XGBoost/
│   ├── xgb_train.py       # Train XGBoost  classification model 
│   └── xgb_shap.py        # SHAP analyze with XGBoost classification model
│
├── addnoise.py            # Injecting noise into classification dataset

Regression task/                  
├── randomforest.py        # Train Random Forest regression model and SHAP analyze
├── xgboost.py             # Train XGBoost regression model and SHAP analyze
└── addnoise.py            # Injecting noise into regression dataset

data/                             
├── Training.csv          # Classification data 
├── Testing.csv           # Classification data
└── global_cancer_patients_2015_2024.csv  # Regression data

README.md                             
.gitignore                            
```
## Datasets
Disease Prediction Using Machine Learning (Kaggle, Kaushil Patel, 2019)
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

Global Cancer Patients 2015–2024 (Kaggle, Zahid Feroze, 2025)
https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024
