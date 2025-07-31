import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.feature_selector = None
        self.polynomial_features = None
    
    def create_derived_features(self, df):
        """Create derived financial features"""
        logger.info("Creating derived features...")
        
        df_features = df.copy()
        
        # Debt-to-income ratio
        df_features['debt_to_income_ratio'] = (
            df_features['monthly_debt_payments'] * 12 / df_features['annual_income']
        )
        
        # Income-to-loan ratio
        df_features['income_to_loan_ratio'] = (
            df_features['annual_income'] / df_features['loan_amount']
        )
        
        # Credit density (accounts per year of employment)
        df_features['credit_density'] = (
            df_features['num_credit_accounts'] / 
            np.maximum(df_features['employment_length'], 1)
        )
        
        # Financial stability score (composite metric)
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
        
        # Loan burden (loan amount relative to annual income)
        df_features['loan_burden'] = (
            df_features['loan_amount'] / df_features['annual_income']
        )
        
        # Credit experience (years of credit history proxy)
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
        
        # Replace inf values with median
        for col in df_features.columns:
            if df_features[col].dtype in ['float64', 'int64']:
                df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
                if df_features[col].isna().any():
                    df_features[col] = df_features[col].fillna(df_features[col].median())
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def create_polynomial_features(self, df, degree=2, interaction_only=True):
        """Create polynomial and interaction features"""
        logger.info("Creating polynomial features...")
        
        # Select only numerical features for polynomial transformation
        numerical_cols = [col for col in df.columns 
                         if df[col].dtype in ['float64', 'int64'] 
                         and col != 'default_risk'][:10]  # Limit to prevent explosion
        
        if self.polynomial_features is None:
            self.polynomial_features = PolynomialFeatures(
                degree=degree, 
                interaction_only=interaction_only,
                include_bias=False
            )
            poly_features = self.polynomial_features.fit_transform(df[numerical_cols])
        else:
            poly_features = self.polynomial_features.transform(df[numerical_cols])
        
        # Create feature names
        feature_names = self.polynomial_features.get_feature_names_out(numerical_cols)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Combine with original data (excluding the original numerical columns to avoid duplication)
        result_df = pd.concat([
            df.drop(columns=numerical_cols),
            poly_df
        ], axis=1)
        
        logger.info(f"Created {len(feature_names)} polynomial features")
        return result_df
    
    def select_best_features(self, X, y, k=50):
        """Select the best features using statistical tests"""
        logger.info(f"Selecting top {k} features...")
        
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
        else:
            X_selected = self.feature_selector.transform(X)
        
        # Get selected feature names
        feature_mask = self.feature_selector.get_support()
        selected_features = X.columns[feature_mask].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_risk_buckets(self, df):
        """Create risk bucket features"""
        logger.info("Creating risk buckets...")
        
        df_buckets = df.copy()
        
        # Income buckets
        df_buckets['income_bucket'] = pd.cut(
            df_buckets['annual_income'],
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        ).astype(str)
        
        # Age buckets
        df_buckets['age_bucket'] = pd.cut(
            df_buckets['age'],
            bins=[0, 25, 35, 45, 55, float('inf')],
            labels=['Young', 'Early Career', 'Mid Career', 'Late Career', 'Senior']
        ).astype(str)
        
        # Loan amount buckets
        df_buckets['loan_bucket'] = pd.cut(
            df_buckets['loan_amount'],
            bins=[0, 10000, 25000, 50000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        ).astype(str)
        
        return df_buckets
    
    def engineer_features(self, df, target_col='default_risk', create_poly=False):
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
        
        # Create derived features
        X_derived = self.create_derived_features(X)
        
        # Create risk buckets
        X_buckets = self.create_risk_buckets(X_derived)
        
        # Create polynomial features if requested
        if create_poly:
            X_poly = self.create_polynomial_features(X_buckets)
        else:
            X_poly = X_buckets
        
        # Feature selection (only if we have target variable)
        if y is not None and len(X_poly.columns) > 100:
            # Encode categorical columns for feature selection
            X_encoded = X_poly.copy()
            categorical_cols = X_encoded.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
            
            X_selected = self.select_best_features(X_encoded, y, k=50)
            result = pd.concat([X_selected, y], axis=1)
        else:
            result = pd.concat([X_poly, y], axis=1) if y is not None else X_poly
        
        logger.info("Feature engineering completed!")
        logger.info(f"Final feature count: {len(result.columns) - (1 if y is not None else 0)}")
        
        return result
    
    def get_feature_importance_summary(self, X, y):
        """Get summary of most important features"""
        if self.feature_selector is not None:
            scores = self.feature_selector.scores_
            feature_names = X.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_score': scores
            }).sort_values('importance_score', ascending=False)
            
            return importance_df
        else:
            logger.warning("Feature selector not fitted. Run feature selection first.")
            return None

def main():
    """Test feature engineering"""
    from config import FEATURE_CONFIG, PROCESSED_DATA_DIR
    
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(FEATURE_CONFIG)
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    # Save engineered features
    output_path = PROCESSED_DATA_DIR / "engineered_features.csv"
    df_engineered.to_csv(output_path, index=False)
    logger.info(f"Engineered features saved to {output_path}")
    
    # Print feature summary
    if 'default_risk' in df_engineered.columns:
        X = df_engineered.drop(columns=['default_risk'])
        y = df_engineered['default_risk']
        
        importance_summary = engineer.get_feature_importance_summary(X, y)
        if importance_summary is not None:
            print("\nTop 10 Most Important Features:")
            print(importance_summary.head(10))

if __name__ == "__main__":
    main()