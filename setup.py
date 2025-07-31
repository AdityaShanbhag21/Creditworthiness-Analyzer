#!/usr/bin/env python3
"""
Quick setup script for AI-Powered Creditworthiness Analyzer
Creates directory structure and runs initial setup
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directory_structure():
    """Create the required directory structure"""
    
    directories = [
        "data/raw",
        "data/processed", 
        "src",
        "models/trained_models",
        "dashboard",
        "notebooks"
    ]
    
    print("📁 Creating directory structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Directory structure created!")

def install_requirements():
    """Install required packages"""
    
    print("\n📦 Installing Python packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def create_sample_notebook():
    """Create a sample Jupyter notebook"""
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🏦 Credit Risk Analysis - Exploratory Data Analysis\\n",
    "\\n",
    "This notebook provides exploratory data analysis for the creditworthiness dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "print('📊 Credit Risk Analysis Notebook Ready!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the processed data\\n",
    "df = pd.read_csv('../data/processed/sample_data.csv')\\n",
    "print(f'Dataset shape: {df.shape}')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic statistics\\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target variable distribution\\n",
    "plt.figure(figsize=(8, 6))\\n",
    "df['default_risk'].value_counts().plot(kind='bar')\\n",
    "plt.title('Default Risk Distribution')\\n",
    "plt.xlabel('Default Risk (0: Low, 1: High)')\\n",
    "plt.ylabel('Count')\\n",
    "plt.xticks(rotation=0)\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    notebook_path = Path("notebooks/exploratory_analysis.ipynb")
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print("📓 Sample notebook created: notebooks/exploratory_analysis.ipynb")

def run_quick_test():
    """Run a quick test to verify setup"""
    
    print("\n🧪 Running quick setup test...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        print("   ✅ Core packages imported successfully")
        
        # Test directory access
        from pathlib import Path
        assert Path("data").exists()
        assert Path("src").exists()
        assert Path("models").exists()
        print("   ✅ Directory structure verified")
        
        print("✅ Setup test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🏦 AI-Powered Creditworthiness Analyzer - Setup")
    print("=" * 60)
    
    # Create directories
    create_directory_structure()
    
    # Install requirements
    if Path("requirements.txt").exists():
        success = install_requirements()
        if not success:
            print("\n⚠️  Package installation failed. You may need to install manually.")
    else:
        print("\n⚠️  requirements.txt not found. Skipping package installation.")
    
    # Create sample notebook
    create_sample_notebook()
    
    # Run test
    test_passed = run_quick_test()
    
    print("\n" + "=" * 60)
    if test_passed:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\n🚀 Next Steps:")
        print("1. Run the pipeline: python run_pipeline.py")
        print("2. Launch dashboard: streamlit run dashboard/streamlit_app.py")
        print("3. Open Jupyter notebook: jupyter notebook notebooks/exploratory_analysis.ipynb")
    else:
        print("⚠️  Setup completed with warnings. Check error messages above.")
    
    print("\n📚 Documentation: README.md")
    print("🐛 Issues: Check pipeline.log for detailed logs")

if __name__ == "__main__":
    main()