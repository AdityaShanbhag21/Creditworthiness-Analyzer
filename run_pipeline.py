#!/usr/bin/env python3
"""
Complete pipeline runner for AI-Powered Creditworthiness Analyzer
Executes the entire ML pipeline from data generation to model training
"""

import sys
import logging
from pathlib import Path
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    
    print("🏦 AI-Powered Creditworthiness Analyzer Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preprocessing
        print("\n📊 Step 1: Data Generation and Preprocessing")
        print("-" * 40)
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        print("✅ Data preprocessing completed successfully!")
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Feature Engineering")
        print("-" * 40)
        from feature_engineering import main as feature_main
        feature_main()
        print("✅ Feature engineering completed successfully!")
        
        # Step 3: Model Training
        print("\n🧠 Step 3: Model Training")
        print("-" * 40)
        from model_training import main as training_main
        training_main()
        print("✅ Model training completed successfully!")
        
        # Step 4: Model Evaluation
        print("\n📈 Step 4: Model Evaluation")
        print("-" * 40)
        from model_evaluation import main as evaluation_main
        evaluation_main()
        print("✅ Model evaluation completed successfully!")
        
        # Step 5: Test Prediction Service
        print("\n🔮 Step 5: Testing Prediction Service")
        print("-" * 40)
        from prediction_service import main as prediction_main
        prediction_main()
        print("✅ Prediction service test completed successfully!")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📁 Models saved in: models/trained_models/")
        print(f"📊 Data saved in: data/processed/")
        print(f"📋 Logs saved in: pipeline.log")
        
        print("\n🚀 Next Steps:")
        print("1. Launch the dashboard: streamlit run dashboard/streamlit_app.py")
        print("2. Open your browser to: http://localhost:8501")
        print("3. Start analyzing creditworthiness!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed at step: {str(e)}")
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("Check pipeline.log for detailed error information")
        return False

def run_individual_step(step_name):
    """Run individual pipeline steps"""
    
    steps = {
        'preprocess': ('Data Preprocessing', 'data_preprocessing'),
        'features': ('Feature Engineering', 'feature_engineering'),
        'train': ('Model Training', 'model_training'),
        'evaluate': ('Model Evaluation', 'model_evaluation'),
        'test': ('Prediction Service Test', 'prediction_service')
    }
    
    if step_name not in steps:
        print(f"❌ Unknown step: {step_name}")
        print(f"Available steps: {', '.join(steps.keys())}")
        return False
    
    step_title, module_name = steps[step_name]
    
    print(f"🔧 Running: {step_title}")
    print("-" * 40)
    
    try:
        module = __import__(module_name)
        module.main()
        print(f"✅ {step_title} completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"{step_title} failed: {str(e)}")
        print(f"❌ {step_title} failed: {str(e)}")
        return False

def check_environment():
    """Check if the environment is properly set up"""
    
    print("🔍 Checking Environment Setup")
    print("-" * 40)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 
        'matplotlib', 'seaborn', 'streamlit', 'plotly', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n❌ Missing packages detected!")
        print("Install with: pip install", " ".join(missing_packages))
        return False
    
    # Check directory structure
    required_dirs = ['data', 'src', 'models', 'dashboard']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory")
        else:
            print(f"❌ {dir_name}/ directory (missing)")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created {dir_name}/ directory")
    
    print("\n✅ Environment check completed!")
    return True

def main():
    """Main function with command line interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI-Powered Creditworthiness Analyzer Pipeline Runner"
    )
    
    parser.add_argument(
        '--step', 
        choices=['preprocess', 'features', 'train', 'evaluate', 'test', 'all'],
        default='all',
        help='Pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='Check environment setup before running'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment checks'
    )
    
    args = parser.parse_args()
    
    # Environment check
    if args.check_env or not args.skip_checks:
        if not check_environment():
            print("\n❌ Environment setup incomplete. Fix issues and try again.")
            return False
    
    # Run pipeline
    if args.step == 'all':
        success = run_complete_pipeline()
    else:
        success = run_individual_step(args.step)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)