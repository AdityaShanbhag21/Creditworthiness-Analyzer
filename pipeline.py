import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv('data/loan_approval_dataset.csv')
    
    # CLEANING: Remove spaces from column names and string values
    df.columns = df.columns.str.strip()
    
    # Force clean strings for categorical data to prevent mapping errors
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    #  FEATURE ENGINEERING
    df['total_assets'] = (df['residential_assets_value'] + 
                          df['commercial_assets_value'] + 
                          df['luxury_assets_value'] + 
                          df['bank_asset_value'])

    #  Encoding Categorical Data
    encoders = {}
    for col in ['education', 'self_employed']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    #  Define Features
    num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'total_assets']
    cat_cols = ['education', 'self_employed']
    
    X = df[num_cols + cat_cols]
    
    # 'Approved' is 1 and 'Rejected' is 0
    df['loan_status'] = df['loan_status'].str.strip()
    y = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Fill any NaNs that might have appeared during mapping
    y = y.fillna(0) 
    
    return X, y, encoders

def get_model(model_name):
    if model_name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return RandomForestClassifier(n_estimators=50)

    
