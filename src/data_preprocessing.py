import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, n_samples=10000):
        """Generate synthetic credit data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'annual_income': np.random.lognormal(10.5, 0.8, n_samples),
            'employment_status': np.random.choice(
                ['Employed', 'Self-employed', 'Unemployed', 'Student'], 
                n_samples, p=[0.6, 0.25, 0.1, 0.05]
            ),
            'employment_length': np.random.exponential(5, n_samples),
            'education_level': np.random.choice(
                ['High School', 'Bachelor', 'Master', 'PhD'], 
                n_samples, p=[0.3, 0.4, 0.25, 0.05]
            ),
            'home_ownership': np.random.choice(
                ['Own', 'Rent', 'Mortgage'], 
                n_samples, p=[0.3, 0.4, 0.3]
            ),
            'loan_amount': np.random.lognormal(9.5, 0.7, n_samples),
            'loan_purpose': np.random.choice(
                ['Home', 'Auto', 'Personal', 'Education', 'Business'], 
                n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            'monthly_debt_payments': np.random.lognormal(7.5, 0.8, n_samples),
            'credit_utilization_ratio': np.random.beta(2, 5, n_samples),
            'num_credit_accounts': np.random.poisson(4, n_samples),
            'num_previous_loans': np.random.poisson(2, n_samples),
            'marital_status': np.random.choice(
                ['Single', 'Married', 'Divorced'], 
                n_samples, p=[0.4, 0.45, 0.15]
            )
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic constraints
        df['age'] = np.clip(df['age'], 18, 80)
        df['annual_income'] = np.clip(df['annual_income'], 15000, 500000)
        df['employment_length'] = np.clip(df['employment_length'], 0, 40)
        df['loan_amount'] = np.clip(df['loan_amount'], 1000, 100000)
        df['monthly_debt_payments'] = np.clip(df['monthly_debt_payments'], 0, df['annual_income']/12 * 0.8)
        df['credit_utilization_ratio'] = np.clip(df['credit_utilization_ratio'], 0, 1)
        df['num_credit_accounts'] = np.clip(df['num_credit_accounts'], 0, 20)
        df['num_previous_loans'] = np.clip(df['num_previous_loans'], 0, 10)
        
        # Generate target variable based on realistic factors
        risk_score = (
            (df['annual_income'] < 30000) * 0.3 +
            (df['employment_status'] == 'Unemployed') * 0.4 +
            (df['credit_utilization_ratio'] > 0.8) * 0.3 +
            (df['monthly_debt_payments'] / (df['annual_income']/12) > 0.4) * 0.3 +
            (df['age'] < 25) * 0.1 +
            (df['num_previous_loans'] > 5) * 0.2 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['default_risk'] = (risk_score > 0.5).astype(int)
        
        return df
    
    def clean_data(self, df):
        """Clean and validate the dataset"""
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        df = df.dropna()
        
        # Remove outliers using IQR method for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'default_risk':  # Don't remove outliers from target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df_encoded[col] = df[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder found for column: {col}")
        
        return df_encoded
    
    def scale_numerical_features(self, df, fit=True):
        """Scale numerical features"""
        logger.info("Scaling numerical features...")
        
        numerical_cols = self.config['numerical_features']
        df_scaled = df.copy()
        
        if fit:
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_scaled
    
    def preprocess_data(self, df, fit=True):
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, fit=fit)
        
        # Scale numerical features
        df_processed = self.scale_numerical_features(df_encoded, fit=fit)
        
        logger.info("Preprocessing completed successfully!")
        return df_processed
    
    def save_preprocessors(self, path):
        """Save preprocessing objects"""
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        joblib.dump(preprocessors, path)
        logger.info(f"Preprocessors saved to {path}")
    
    def load_preprocessors(self, path):
        """Load preprocessing objects"""
        preprocessors = joblib.load(path)
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        logger.info(f"Preprocessors loaded from {path}")

def main():
    """Generate and preprocess sample data"""
    from config import FEATURE_CONFIG, PROCESSED_DATA_DIR
    
    preprocessor = DataPreprocessor(FEATURE_CONFIG)
    
    # Generate sample data
    logger.info("Generating sample data...")
    df = preprocessor.generate_sample_data(10000)
    
    # Save raw data
    raw_data_path = PROCESSED_DATA_DIR / "sample_data.csv"
    df.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    # Preprocess data
    df_processed = preprocessor.preprocess_data(df, fit=True)
    
    # Save processed data
    processed_data_path = PROCESSED_DATA_DIR / "processed_data.csv"
    df_processed.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Save preprocessors
    preprocessors_path = PROCESSED_DATA_DIR / "preprocessors.joblib"
    preprocessor.save_preprocessors(preprocessors_path)

if __name__ == "__main__":
    main()