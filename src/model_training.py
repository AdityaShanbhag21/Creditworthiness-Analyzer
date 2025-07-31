import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditworthinessModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, df, target_col='default_risk', handle_imbalance=True):
        logger.info("Preparing data for training...")
        X = df.drop(columns=[target_col])
        y = df[target_col]

        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True) if len(categorical_cols) > 0 else X.copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        if handle_imbalance:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=self.config['random_state'])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Class distribution: {np.bincount(y_train)}")

        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train):
        logger.info("Training Logistic Regression...")
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        lr = LogisticRegression(max_iter=1000, **self.config['models']['logistic_regression'])

        grid_search = GridSearchCV(
            lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_lr = grid_search.best_estimator_
        self.models['logistic_regression'] = best_lr

        logger.info(f"Best LR parameters: {grid_search.best_params_}")
        return best_lr

    def train_random_forest(self, X_train, y_train):
        logger.info("Training Random Forest...")

        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }

        rf = RandomForestClassifier(**self.config['models']['random_forest'])

        grid_search = GridSearchCV(
            rf, reduced_param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        self.models['random_forest'] = best_rf

        logger.info(f"Best RF parameters: {grid_search.best_params_}")
        return best_rf

    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost...")

        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            **self.config['models']['xgboost']
        )

        grid_search = GridSearchCV(
            xgb_model, reduced_param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_xgb = grid_search.best_estimator_
        self.models['xgboost'] = best_xgb

        logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        return best_xgb

    def evaluate_model(self, model, X_test, y_test, model_name):
        logger.info(f"Evaluating {model_name}...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        self.model_scores[model_name] = {
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        logger.info(f"{model_name} AUC Score: {auc_score:.4f}")
        return auc_score

    def train_all_models(self, X_train, y_train, X_test, y_test):
        logger.info("Training all models...")

        models_to_train = {
            'logistic_regression': self.train_logistic_regression,
            'random_forest': self.train_random_forest,
            'xgboost': self.train_xgboost
        }

        for model_name, train_func in models_to_train.items():
            try:
                model = train_func(X_train, y_train)
                self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")

        best_auc = 0
        for model_name, scores in self.model_scores.items():
            if scores['auc_score'] > best_auc:
                best_auc = scores['auc_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]

        logger.info(f"Best model: {self.best_model_name} (AUC: {best_auc:.4f})")
        return self.models, self.model_scores

    def cross_validate_models(self, X_train, y_train):
        logger.info("Performing cross-validation...")
        cv_scores = {}

        for model_name, model in self.models.items():
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['cross_validation_folds'],
                scoring='roc_auc'
            )
            cv_scores[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            logger.info(f"{model_name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_scores

    def get_feature_importance(self, model_name=None):
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)

        if model is None:
            logger.warning(f"Model {model_name} not found")
            return None

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} doesn't have feature importance")
            return None

    def plot_model_comparison(self, save_path=None):
        if not self.model_scores:
            logger.warning("No model scores available for plotting")
            return

        models = list(self.model_scores.keys())
        auc_scores = [self.model_scores[model]['auc_score'] for model in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Performance Comparison (AUC Score)', fontsize=16, fontweight='bold')
        plt.ylabel('AUC Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)

        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        plt.show()

    def plot_roc_curves(self, X_test, y_test, save_path=None):
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i, (model_name, model) in enumerate(self.models.items()):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = self.model_scores[model_name]['auc_score']
                plt.plot(fpr, tpr, color=colors[i % len(colors)],
                          label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {save_path}")
        plt.show()

    def save_models(self, models_dir):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")

        metadata = {
            'model_scores': self.model_scores,
            'best_model_name': self.best_model_name,
            'config': self.config
        }
        metadata_path = models_dir / "models_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")

    def load_models(self, models_dir):
        models_dir = Path(models_dir)
        metadata_path = models_dir / "models_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_scores = metadata['model_scores']
            self.best_model_name = metadata['best_model_name']
            self.config = metadata['config']

        for model_file in models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace('_model', '')
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} from {model_file}")

        if self.best_model_name:
            self.best_model = self.models[self.best_model_name]
            logger.info(f"Loaded best model: {self.best_model_name}")
        else:
            logger.warning("No best model found in metadata")
            self.best_model = None

def main():
    """Main training pipeline"""
    from config import MODEL_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR

    # Load engineered features
    df = pd.read_csv(PROCESSED_DATA_DIR / "engineered_features.csv")

    # Initialize trainer
    trainer = CreditworthinessModelTrainer(MODEL_CONFIG)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    # Train all models
    models, scores = trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Cross-validate models
    cv_scores = trainer.cross_validate_models(X_train, y_train)

    # Plot results
    trainer.plot_model_comparison()
    trainer.plot_roc_curves(X_test, y_test)

    # Save models
    trainer.save_models(MODELS_DIR)

    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Best AUC Score: {trainer.model_scores[trainer.best_model_name]['auc_score']:.4f}")

    print("\nAll Model Scores:")
    for model_name, score_data in trainer.model_scores.items():
        print(f"{model_name}: {score_data['auc_score']:.4f}")

    print("\nCross-Validation Scores:")
    for model_name, cv_data in cv_scores.items():
        print(f"{model_name}: {cv_data['mean_score']:.4f} (+/- {cv_data['std_score'] * 2:.4f})")

if __name__ == "__main__":
    main()

                