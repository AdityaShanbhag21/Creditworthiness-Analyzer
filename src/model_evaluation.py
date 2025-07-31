import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.evaluation_results = {}
    
    def load_models(self, models_dir):
        """Load trained models"""
        models_dir = Path(models_dir)
        
        # Load metadata
        metadata_path = models_dir / "models_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_scores = metadata.get('model_scores', {})
        
        # Load models
        for model_file in models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace('_model', '')
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name}")
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Perform comprehensive evaluation of all models"""
        logger.info("Starting comprehensive evaluation...")
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Store results
            self.evaluation_results[model_name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics
            }
        
        return self.evaluation_results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def plot_confusion_matrices(self, y_test, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = confusion_matrix(y_test, results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_test, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                auc_score = results['metrics']['auc_score']
                
                plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                        label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, save_path=None):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if results['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(y_test, results['probabilities'])
                avg_precision = results['metrics']['avg_precision']
                
                plt.plot(recall, precision, color=colors[i % len(colors)], 
                        label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot comprehensive metrics comparison"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
        
        # Prepare data
        model_names = list(self.evaluation_results.keys())
        metrics_data = {}
        
        for metric in metrics_to_plot:
            metrics_data[metric] = [
                self.evaluation_results[model]['metrics'][metric] 
                for model in model_names
            ]
        
        # Create subplot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightsteelblue']
        
        for i, metric in enumerate(metrics_to_plot):
            bars = axes[i].bar(model_names, metrics_data[metric], color=colors)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metrics_data[metric]):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AUC Score comparison (if available)
        if any('auc_score' in self.evaluation_results[model]['metrics'] 
               for model in model_names):
            auc_scores = [
                self.evaluation_results[model]['metrics'].get('auc_score', 0)
                for model in model_names
            ]
            bars = axes[5].bar(model_names, auc_scores, color='gold')
            axes[5].set_title('AUC Score', fontweight='bold')
            axes[5].set_ylim(0, 1)
            axes[5].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, auc_scores):
                axes[5].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(self, save_path=None):
        """Create comprehensive evaluation report"""
        report = {
            'model_comparison': {},
            'detailed_metrics': {},
            'recommendations': {}
        }
        
        # Model comparison
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            report['model_comparison'][model_name] = {
                'AUC Score': f"{metrics.get('auc_score', 0):.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}"
            }
            
            report['detailed_metrics'][model_name] = metrics
        
        # Find best model for different criteria
        best_auc = max(self.evaluation_results.items(), 
                      key=lambda x: x[1]['metrics'].get('auc_score', 0))
        best_precision = max(self.evaluation_results.items(), 
                           key=lambda x: x[1]['metrics']['precision'])
        best_recall = max(self.evaluation_results.items(), 
                         key=lambda x: x[1]['metrics']['recall'])
        
        report['recommendations'] = {
            'best_overall_model': best_auc[0],
            'best_precision_model': best_precision[0],
            'best_recall_model': best_recall[0],
            'model_selection_guidance': {
                'Conservative Lending': 'Use model with highest precision to minimize false positives',
                'Inclusive Lending': 'Use model with highest recall to minimize false negatives',
                'Balanced Approach': 'Use model with highest AUC/F1 score for overall performance'
            }
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def analyze_prediction_distribution(self, save_path=None):
        """Analyze prediction probability distributions"""
        fig, axes = plt.subplots(1, len(self.models), figsize=(5*len(self.models), 4))
        
        if len(self.models) == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if results['probabilities'] is not None:
                axes[i].hist(results['probabilities'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
                axes[i].set_title(f'{model_name}\nPrediction Probability Distribution')
                axes[i].set_xlabel('Predicted Probability of Default')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distribution plot saved to {save_path}")
        
        plt.show()
    
    def print_evaluation_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Create comparison table
        df_comparison = pd.DataFrame()
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            df_comparison[model_name] = [
                f"{metrics.get('auc_score', 0):.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['specificity']:.4f}"
            ]
        
        df_comparison.index = ['AUC Score', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']
        print(df_comparison)
        
        # Best model recommendations
        print("\n" + "-"*50)
        print("RECOMMENDATIONS")
        print("-"*50)
        
        best_auc = max(self.evaluation_results.items(), 
                      key=lambda x: x[1]['metrics'].get('auc_score', 0))
        print(f"Best Overall Model (AUC): {best_auc[0]} ({best_auc[1]['metrics'].get('auc_score', 0):.4f})")
        
        best_precision = max(self.evaluation_results.items(), 
                           key=lambda x: x[1]['metrics']['precision'])
        print(f"Best Precision Model: {best_precision[0]} ({best_precision[1]['metrics']['precision']:.4f})")
        
        best_recall = max(self.evaluation_results.items(), 
                         key=lambda x: x[1]['metrics']['recall'])
        print(f"Best Recall Model: {best_recall[0]} ({best_recall[1]['metrics']['recall']:.4f})")

def main():
    """Main evaluation pipeline"""
    from config import MODELS_DIR, PROCESSED_DATA_DIR
    
    # Load test data
    df = pd.read_csv(PROCESSED_DATA_DIR / "engineered_features.csv")
    X = df.drop(columns=['default_risk'])
    y = df['default_risk']
    
    # For evaluation, we'll use a portion of the data as test set
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle categorical variables
    categorical_cols = X_test.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models(MODELS_DIR)
    
    # Comprehensive evaluation
    evaluator.comprehensive_evaluation(X_test, y_test)
    
    # Generate plots
    evaluator.plot_confusion_matrices(y_test)
    evaluator.plot_roc_curves(y_test)
    evaluator.plot_precision_recall_curves(y_test)
    evaluator.plot_metrics_comparison()
    evaluator.analyze_prediction_distribution()
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(MODELS_DIR / "evaluation_report.json")
    
    # Print summary
    evaluator.print_evaluation_summary()

if __name__ == "__main__":
    main()