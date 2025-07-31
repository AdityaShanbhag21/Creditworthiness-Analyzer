"""
AI-Powered Creditworthiness Analyzer
====================================

A comprehensive fintech application for credit risk assessment using machine learning.

Modules:
--------
- data_preprocessing: Data cleaning and preprocessing utilities
- feature_engineering: Feature creation and selection tools
- model_training: Machine learning model training pipeline
- model_evaluation: Model performance evaluation and metrics
- prediction_service: Prediction API and inference service

Usage:
------
>>> from src.prediction_service import CreditworthinessPredictionService
>>> service = CreditworthinessPredictionService("models/trained_models")
>>> result = service.predict_single(applicant_data)
"""

__version__ = "1.0.0"
__author__ = "AI-Powered Creditworthiness Analyzer Team"
__email__ = "support@creditanalyzer.ai"

# Import main classes for easy access
try:
    from .data_preprocessing import DataPreprocessor
    from .feature_engineering import FeatureEngineer
    from .model_training import CreditworthinessModelTrainer
    from .model_evaluation import ModelEvaluator
    from .prediction_service import CreditworthinessPredictionService
    
    __all__ = [
        'DataPreprocessor',
        'FeatureEngineer', 
        'CreditworthinessModelTrainer',
        'ModelEvaluator',
        'CreditworthinessPredictionService'
    ]
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)
    __all__ = []
