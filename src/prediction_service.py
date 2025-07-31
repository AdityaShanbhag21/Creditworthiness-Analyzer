import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditworthinessPredictionService:
    def __init__(self, models_dir: str, preprocessors_path: str = None):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessors = None
        self.feature_columns = None
        self.best_model_name = None
        
        # Load models and preprocessors
        self._load_models()
        if preprocessors_path:
            self._load_preprocessors(preprocessors_path)
    
    def _load_models(self):
        """Load all trained models"""
        try:
            # Load metadata
            metadata_path = self.models_dir / "models_metadata.joblib"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                self.best_model_name = metadata.get('best_model_name')
                logger.info(f"Best model: {self.best_model_name}")
            
            # Load individual models
            for model_file in self.models_dir.glob("*_model.joblib"):
                model_name = model_file.stem.replace('_model', '')
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_name}")
            
            if not self.models:
                raise FileNotFoundError("No models found in the specified directory")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_preprocessors(self, preprocessors_path: str):
        """Load preprocessing objects"""
        try:
            self.preprocessors = joblib.load(preprocessors_path)
            logger.info("Preprocessors loaded successfully")
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
    
    def _preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing if available
        if self.preprocessors:
            # Encode categorical features
            label_encoders = self.preprocessors.get('label_encoders', {})
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        # Handle unseen categories
                        logger.warning(f"Unseen category in {col}, using default encoding")
                        df[col] = -1
            
            # Scale numerical features
            scaler = self.preprocessors.get('scaler')
            if scaler and hasattr(scaler, 'transform'):
                numerical_cols = [col for col in df.columns 
                                if df[col].dtype in ['float64', 'int64']]
                df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to input data"""
        df_features = df.copy()
        
        # Create derived features (same as in feature_engineering.py)
        try:
            # Debt-to-income ratio
            df_features['debt_to_income_ratio'] = (
                df_features['monthly_debt_payments'] * 12 / df_features['annual_income']
            )
            
            # Income-to-loan ratio
            df_features['income_to_loan_ratio'] = (
                df_features['annual_income'] / df_features['loan_amount']
            )
            
            # Credit density
            df_features['credit_density'] = (
                df_features['num_credit_accounts'] / 
                np.maximum(df_features['employment_length'], 1)
            )
            
            # Financial stability score
            df_features['financial_stability_score'] = (
                (df_features['annual_income'] / 50000) * 0.3 +
                (1 - df_features['credit_utilization_ratio']) * 0.2 +
                (df_features['employment_length'] / 10) * 0.2 +
                (1 / (1 + df_features['debt_to_income_ratio'])) * 0.3
            )
            
            # Age-income interaction
            df_features['age_income_interaction'] = (
                df_features['age'] * df_features['annual_income'] / 1000000
            )
            
            # Loan burden
            df_features['loan_burden'] = (
                df_features['loan_amount'] / df_features['annual_income']
            )
            
            # Credit experience
            df_features['credit_experience'] = np.maximum(
                df_features['age'] - 18, 0
            ) * (df_features['num_credit_accounts'] / 10)
            
            # Risk indicators
            df_features['high_utilization'] = (
                df_features['credit_utilization_ratio'] > 0.7
            ).astype(int)
            
            df_features['high_debt_ratio'] = (
                df_features['debt_to_income_ratio'] > 0.4
            ).astype(int)
            
            df_features['young_borrower'] = (
                df_features['age'] < 25
            ).astype(int)
            
            # Risk buckets
            df_features['income_bucket'] = pd.cut(
                df_features['annual_income'],
                bins=[0, 30000, 50000, 75000, 100000, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            ).astype(str)
            
            df_features['age_bucket'] = pd.cut(
                df_features['age'],
                bins=[0, 25, 35, 45, 55, float('inf')],
                labels=['Young', 'Early Career', 'Mid Career', 'Late Career', 'Senior']
            ).astype(str)
            
            df_features['loan_bucket'] = pd.cut(
                df_features['loan_amount'],
                bins=[0, 10000, 25000, 50000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Very Large']
            ).astype(str)
            
            # Handle categorical variables
            categorical_cols = df_features.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df_features = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
            
            # Replace inf values with median
            for col in df_features.columns:
                if df_features[col].dtype in ['float64', 'int64']:
                    df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
                    if df_features[col].isna().any():
                        df_features[col] = df_features[col].fillna(df_features[col].median())
            
        except Exception as e:
            logger.warning(f"Error in feature engineering: {str(e)}")
        
        return df_features
    
    def predict_single(self, 
                      input_data: Dict, 
                      model_name: Optional[str] = None) -> Dict:
        """Make prediction for a single applicant"""
        
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name or list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Preprocess input
            df = self._preprocess_input(input_data)
            
            # Engineer features
            df_features = self._engineer_features(df)
            
            # Align features with model's expected features
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                missing_features = set(expected_features) - set(df_features.columns)
                extra_features = set(df_features.columns) - set(expected_features)
                
                # Add missing features with zeros
                for feature in missing_features:
                    df_features[feature] = 0
                
                # Remove extra features
                df_features = df_features[expected_features]
            
            # Make predictions
            prediction = model.predict(df_features)[0]
            prediction_proba = model.predict_proba(df_features)[0]
            
            # Calculate risk level
            risk_probability = prediction_proba[1]
            
            if risk_probability < 0.3:
                risk_level = "Low"
                risk_color = "#2ecc71"
            elif risk_probability < 0.7:
                risk_level = "Medium"
                risk_color = "#f39c12"
            else:
                risk_level = "High"
                risk_color = "#e74c3c"
            
            # Generate explanation
            explanation = self._generate_explanation(input_data, risk_probability)
            
            result = {
                'prediction': int(prediction),
                'risk_probability': float(risk_probability),
                'approval_probability': float(1 - risk_probability),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'model_used': model_name,
                'explanation': explanation,
                'recommendation': self._generate_recommendation(risk_probability)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, 
                     input_data_list: List[Dict], 
                     model_name: Optional[str] = None) -> List[Dict]:
        """Make predictions for multiple applicants"""
        results = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                result = self.predict_single(input_data, model_name)
                result['applicant_id'] = i + 1
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for applicant {i+1}: {str(e)}")
                results.append({
                    'applicant_id': i + 1,
                    'error': str(e),
                    'prediction': None
                })
        
        return results
    
    def _generate_explanation(self, input_data: Dict, risk_probability: float) -> List[str]:
        """Generate human-readable explanation for the prediction"""
        explanations = []
        
        # Income analysis
        annual_income = input_data.get('annual_income', 0)
        if annual_income < 30000:
            explanations.append("⚠️ Low annual income increases credit risk")
        elif annual_income > 75000:
            explanations.append("✅ High annual income reduces credit risk")
        
        # Debt-to-income analysis
        monthly_debt = input_data.get('monthly_debt_payments', 0)
        debt_to_income = (monthly_debt * 12) / annual_income if annual_income > 0 else 0
        if debt_to_income > 0.4:
            explanations.append("⚠️ High debt-to-income ratio increases risk")
        elif debt_to_income < 0.2:
            explanations.append("✅ Low debt-to-income ratio is favorable")
        
        # Credit utilization analysis
        credit_util = input_data.get('credit_utilization_ratio', 0)
        if credit_util > 0.8:
            explanations.append("⚠️ High credit utilization increases risk")
        elif credit_util < 0.3:
            explanations.append("✅ Low credit utilization is positive")
        
        # Employment analysis
        employment_status = input_data.get('employment_status', '')
        if employment_status == 'Unemployed':
            explanations.append("⚠️ Unemployment significantly increases risk")
        elif employment_status == 'Employed':
            employment_length = input_data.get('employment_length', 0)
            if employment_length > 5:
                explanations.append("✅ Stable employment history is favorable")
        
        # Age analysis
        age = input_data.get('age', 0)
        if age < 25:
            explanations.append("⚠️ Young age may indicate limited credit history")
        elif age > 40:
            explanations.append("✅ Mature age suggests financial stability")
        
        # Loan amount vs income
        loan_amount = input_data.get('loan_amount', 0)
        loan_to_income = loan_amount / annual_income if annual_income > 0 else 0
        if loan_to_income > 3:
            explanations.append("⚠️ High loan amount relative to income")
        elif loan_to_income < 1:
            explanations.append("✅ Conservative loan amount relative to income")
        
        if not explanations:
            explanations.append("📊 Risk assessment based on overall financial profile")
        
        return explanations
    
    def _generate_recommendation(self, risk_probability: float) -> Dict:
        """Generate lending recommendation"""
        if risk_probability < 0.3:
            return {
                'decision': 'APPROVE',
                'confidence': 'High',
                'suggested_terms': {
                    'interest_rate': 'Standard rates apply',
                    'conditions': 'Standard terms',
                    'monitoring': 'Regular monitoring'
                }
            }
        elif risk_probability < 0.7:
            return {
                'decision': 'CONDITIONAL APPROVAL',
                'confidence': 'Medium',
                'suggested_terms': {
                    'interest_rate': 'Higher interest rate recommended',
                    'conditions': 'Additional collateral or co-signer',
                    'monitoring': 'Enhanced monitoring required'
                }
            }
        else:
            return {
                'decision': 'DECLINE',
                'confidence': 'High',
                'suggested_terms': {
                    'interest_rate': 'N/A',
                    'conditions': 'Consider debt counseling or secured loan',
                    'monitoring': 'Re-evaluate after 6-12 months'
                }
            }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'best_model': self.best_model_name,
            'total_models': len(self.models)
        }
    
    def validate_input(self, input_data: Dict) -> Dict:
        """Validate input data"""
        required_fields = [
            'age', 'annual_income', 'employment_status', 'employment_length',
            'education_level', 'home_ownership', 'loan_amount', 'loan_purpose',
            'monthly_debt_payments', 'credit_utilization_ratio', 'num_credit_accounts',
            'num_previous_loans', 'marital_status'
        ]
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        for field in required_fields:
            if field not in input_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Validate data types and ranges
        if 'age' in input_data:
            age = input_data['age']
            if not isinstance(age, (int, float)) or age < 18 or age > 100:
                validation_result['errors'].append("Age must be between 18 and 100")
                validation_result['is_valid'] = False
        
        if 'annual_income' in input_data:
            income = input_data['annual_income']
            if not isinstance(income, (int, float)) or income < 0:
                validation_result['errors'].append("Annual income must be positive")
                validation_result['is_valid'] = False
            elif income < 15000:
                validation_result['warnings'].append("Very low income detected")
        
        if 'credit_utilization_ratio' in input_data:
            util = input_data['credit_utilization_ratio']
            if not isinstance(util, (int, float)) or util < 0 or util > 1:
                validation_result['errors'].append("Credit utilization ratio must be between 0 and 1")
                validation_result['is_valid'] = False
        
        return validation_result

def main():
    """Test the prediction service"""
    from config import MODELS_DIR, PROCESSED_DATA_DIR
    
    # Initialize prediction service
    preprocessors_path = PROCESSED_DATA_DIR / "preprocessors.joblib"
    service = CreditworthinessPredictionService(MODELS_DIR, preprocessors_path)
    
    # Test sample prediction
    sample_input = {
        'age': 35,
        'annual_income': 65000,
        'employment_status': 'Employed',
        'employment_length': 5.0,
        'education_level': 'Bachelor',
        'home_ownership': 'Mortgage',
        'loan_amount': 25000,
        'loan_purpose': 'Home',
        'monthly_debt_payments': 1500,
        'credit_utilization_ratio': 0.3,
        'num_credit_accounts': 4,
        'num_previous_loans': 2,
        'marital_status': 'Married'
    }
    
    # Validate input
    validation = service.validate_input(sample_input)
    print("Validation Result:", validation)
    
    if validation['is_valid']:
        # Make prediction
        result = service.predict_single(sample_input)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Probability: {result['risk_probability']:.3f}")
        print(f"Approval Probability: {result['approval_probability']:.3f}")
        print(f"Recommendation: {result['recommendation']['decision']}")
        print(f"Model Used: {result['model_used']}")
        
        print("\nExplanation:")
        for explanation in result['explanation']:
            print(f"  {explanation}")
        
        print(f"\nSuggested Terms:")
        terms = result['recommendation']['suggested_terms']
        for key, value in terms.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()