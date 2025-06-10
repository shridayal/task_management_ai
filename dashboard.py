"""Dashboard to display results from all weeks"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results():
    """Load results from all weeks"""
    results = {}
    
    # Check if results files exist
    if Path('results/performance_metrics.csv').exists():
        results['performance'] = pd.read_csv('results/performance_metrics.csv')
    
    if Path('results/feature_importance.csv').exists():
        results['features'] = pd.read_csv('results/feature_importance.csv')
        
    return results

def create_summary_report():
    """Create a summary report of all weeks"""
    
    print("\n" + "="*60)
    print("TASK MANAGEMENT AI - SUMMARY REPORT")
    print("="*60)
    
    # Week 1 Summary
    print("\n📊 WEEK 1 - Data Analysis Summary:")
    print("- Dataset loaded and preprocessed")
    print("- Text preprocessing with NLTK completed")
    print("- Feature engineering completed")
    
    # Week 2 Summary  
    print("\n🤖 WEEK 2 - Basic Models Summary:")
    print("- TF-IDF features extracted")
    print("- Naive Bayes model trained")
    print("- SVM model trained")
    
    # Week 3 Summary
    print("\n🚀 WEEK 3 - Advanced Models Summary:")
    print("- Random Forest with GridSearchCV")
    print("- XGBoost with hyperparameter tuning")
    print("- Workload balancing implemented")
    
    # Week 4 Summary
    print("\n✅ WEEK 4 - Final Results:")
    results = load_results()
    
    if 'performance' in results:
        print("\nModel Performance:")
        print(results['performance'])
    
    print("\n" + "="*60)

if __name__ == "__main__":
    create_summary_report()
