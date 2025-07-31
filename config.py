import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'models': {
        'logistic_regression': {
            'max_iter': 1000,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 100,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'age', 'annual_income', 'employment_length', 'loan_amount',
        'monthly_debt_payments', 'credit_utilization_ratio',
        'num_credit_accounts', 'num_previous_loans'
    ],
    'categorical_features': [
        'employment_status', 'education_level', 'home_ownership',
        'loan_purpose', 'marital_status'
    ],
    'derived_features': [
        'debt_to_income_ratio', 'income_to_loan_ratio',
        'credit_density', 'financial_stability_score'
    ]
}

# Risk categories
RISK_CATEGORIES = {
    0: {'label': 'Low Risk', 'color': '#2ecc71', 'description': 'High probability of loan repayment'},
    1: {'label': 'High Risk', 'color': '#e74c3c', 'description': 'Low probability of loan repayment'}
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'AI Creditworthiness Analyzer',
    'page_icon': '💳',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}