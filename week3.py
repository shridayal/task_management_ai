import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('preprocessed_tasks.csv')

# Create additional features for better priority prediction
print("=== Feature Engineering for Priority Prediction ===")

# 1. Text-based features
df['summary_length'] = df['Summary'].str.len()
df['word_count'] = df['Summary'].str.split().str.len()
df['exclamation_count'] = df['Summary'].str.count('!')
df['question_count'] = df['Summary'].str.count('\?')

# 2. Label-based features (encode categorical)
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Label'])

# 3. Assignee workload features
assignee_workload = df.groupby('Assignee').size().to_dict()
df['assignee_current_load'] = df['Assignee'].map(assignee_workload)

# 4. Time-based features
df['Created'] = pd.to_datetime(df['Created'])
df['day_of_week'] = df['Created'].dt.dayofweek
df['month'] = df['Created'].dt.month
df['quarter'] = df['Created'].dt.quarter

# 5. Historical assignee performance
assignee_priority_dist = df.groupby(['Assignee', 'Priority']).size().unstack(fill_value=0)
assignee_high_priority_ratio = (assignee_priority_dist['High'] / assignee_priority_dist.sum(axis=1)).to_dict()
df['assignee_high_priority_ratio'] = df['Assignee'].map(assignee_high_priority_ratio)

# Combine features
feature_columns = ['summary_length', 'word_count', 'exclamation_count', 'question_count',
                  'label_encoded', 'assignee_current_load', 'day_of_week', 'month',
                  'quarter', 'assignee_high_priority_ratio']

print(f"Total features created: {len(feature_columns)}")
print(f"Feature names: {feature_columns}")



# Load TF-IDF features from Week 2
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
tfidf_features = tfidf_vectorizer.transform(df['processed_summary'])

# Combine TF-IDF with engineered features
X_engineered = df[feature_columns].values
X_combined = np.hstack([tfidf_features.toarray(), X_engineered])

print(f"\nCombined feature shape: {X_combined.shape}")

# Prepare target
y = df['Priority']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)



print("\n=== Random Forest with Hyperparameter Tuning ===")

# Define parameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# GridSearchCV
rf_grid_search = GridSearchCV(
    rf, 
    rf_param_grid, 
    cv=5, 
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("Starting Random Forest GridSearchCV...")
rf_grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"\nBest RF Parameters: {rf_grid_search.best_params_}")
print(f"Best RF CV Score: {rf_grid_search.best_score_:.3f}")

# Evaluate on test set
rf_best = rf_grid_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print(f"\nRandom Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


print("\n=== XGBoost with Hyperparameter Tuning ===")

# Encode labels for XGBoost
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_train_xgb = y_train.map(label_mapping)
y_test_xgb = y_test.map(label_mapping)

# Define parameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# GridSearchCV
xgb_grid_search = GridSearchCV(
    xgb,
    xgb_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("Starting XGBoost GridSearchCV...")
xgb_grid_search.fit(X_train, y_train_xgb)

# Best parameters and score
print(f"\nBest XGBoost Parameters: {xgb_grid_search.best_params_}")
print(f"Best XGBoost CV Score: {xgb_grid_search.best_score_:.3f}")

# Evaluate on test set
xgb_best = xgb_grid_search.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

# Convert back to original labels
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_pred_xgb_labels = pd.Series(y_pred_xgb).map(reverse_mapping)

print(f"\nXGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb_labels):.3f}")



print("\n=== ML-based Workload Optimization ===")

# Create a more sophisticated workload balancing model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

class MLWorkloadOptimizer:
    def __init__(self, historical_data):
        self.data = historical_data
        self.model = None
        self.prepare_training_data()
        
    def prepare_training_data(self):
        """Prepare features for workload optimization"""
        # Create historical performance metrics
        assignee_metrics = []
        
        for assignee in self.data['Assignee'].unique():
            assignee_data = self.data[self.data['Assignee'] == assignee]
            
            # Calculate metrics
            metrics = {
                'assignee': assignee,
                'avg_completion_time': np.random.uniform(1, 10),  # Simulated
                'task_success_rate': len(assignee_data[assignee_data['Status'] == 'Done']) / len(assignee_data),
                'high_priority_completion': len(assignee_data[(assignee_data['Priority'] == 'High') & 
                                                             (assignee_data['Status'] == 'Done')]) / 
                                          max(len(assignee_data[assignee_data['Priority'] == 'High']), 1)
            }
            assignee_metrics.append(metrics)
            
        self.assignee_metrics = pd.DataFrame(assignee_metrics)
        
    def create_optimization_features(self, task_features, assignee):
        """Create features for optimization model"""
        assignee_metric = self.assignee_metrics[self.assignee_metrics['assignee'] == assignee].iloc[0]
        
        return np.array([
            task_features[0],  # task complexity
            assignee_metric['task_success_rate'],
            assignee_metric['high_priority_completion'],
            assignee_metric['avg_completion_time']
        ])
    
    def optimize_assignment(self, tasks_batch):
        """Optimize assignment for a batch of tasks"""
        assignments = []
        assignee_loads = {a: 0 for a in self.data['Assignee'].unique()}
        
        # Sort tasks by priority (High -> Medium -> Low)
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        tasks_batch_sorted = sorted(tasks_batch, key=lambda x: priority_order[x['priority']])
        
        for task in tasks_batch_sorted:
            best_score = -float('inf')
            best_assignee = None
            
            for assignee in assignee_loads.keys():
                # Calculate assignment score
                current_load = assignee_loads[assignee]
                assignee_metric = self.assignee_metrics[self.assignee_metrics['assignee'] == assignee].iloc[0]
                
                # Score factors: success rate, current load, priority handling
                score = (assignee_metric['task_success_rate'] * 100 - 
                        current_load * 2 +
                        assignee_metric['high_priority_completion'] * 50 * (1 if task['priority'] == 'High' else 0))
                
                if score > best_score:
                    best_score = score
                    best_assignee = assignee
            
            # Update load
            task_hours = {'Low': 2, 'Medium': 5, 'High': 8}
            assignee_loads[best_assignee] += task_hours[task['priority']]
            
            assignments.append({
                'task': task['summary'],
                'assignee': best_assignee,
                'priority': task['priority'],
                'assignment_score': best_score
            })
            
        return pd.DataFrame(assignments)

# Initialize ML optimizer
ml_optimizer = MLWorkloadOptimizer(df)

# Create batch of tasks for optimization
task_batch = [
    {'summary': 'Critical system update', 'priority': 'High'},
    {'summary': 'Data analysis report', 'priority': 'Medium'},
    {'summary': 'Code refactoring', 'priority': 'Low'},
    {'summary': 'Security vulnerability fix', 'priority': 'High'},
    {'summary': 'Documentation update', 'priority': 'Low'}
]

optimized_assignments = ml_optimizer.optimize_assignment(task_batch)
print("\n=== Optimized Task Assignments ===")
print(optimized_assignments)


# Compare all models
print("\n=== Final Model Comparison ===")

# Load previous models
nb_model = joblib.load('models/naive_bayes_label_classifier.pkl')
svm_model = joblib.load('models/svm_label_classifier.pkl')

# Create comparison dataframe
model_comparison = pd.DataFrame({
    'Model': ['Naive Bayes', 'SVM', 'Random Forest', 'XGBoost'],
    'Task': ['Priority Classification'] * 4,
    'Accuracy': [
        0.85,  # Replace with actual NB priority accuracy from week 2
        0.87,  # Replace with actual SVM priority accuracy from week 2
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb_labels)
    ],
    'Best_Params': [
        'Default',
        'Default',
        str(rf_grid_search.best_params_),
        str(xgb_grid_search.best_params_)
    ]
})

print(model_comparison)

# Visualization of model comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_comparison.plot(x='Model', y='Accuracy', kind='bar', ax=ax, color=['blue', 'green', 'orange', 'red'])
ax.set_title('Model Performance Comparison')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.8, 1.0])
ax.axhline(y=0.9, color='gray', linestyle='--', label='90% threshold')
ax.legend()

# Add value labels on bars
for i, v in enumerate(model_comparison['Accuracy']):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.show()


# Save models and results
print("\n=== Saving Week 3 Results ===")

# Save models
joblib.dump(rf_best, 'models/random_forest_priority_optimized.pkl')
joblib.dump(xgb_best, 'models/xgboost_priority_optimized.pkl')
joblib.dump(balancer, 'models/workload_balancer.pkl')

# Save results
model_comparison.to_csv('results/week3_model_comparison.csv', index=False)
optimized_assignments.to_csv('results/optimized_assignments_sample.csv', index=False)

# Save feature importance
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:20],  # Top 20
    'rf_importance': rf_best.feature_importances_[:20],
    'xgb_importance': xgb_best.feature_importances_[:20]
})
feature_importance_df.to_csv('results/feature_importance.csv', index=False)

# Update README
readme_update = f"""
## Week 3 Progress

### Advanced Models Implemented:
- **Random Forest (Optimized)**
  - Best Parameters: {rf_grid_search.best_params_}
  - Test Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}

- **XGBoost (Optimized)**
  - Best Parameters: {xgb_grid_search.best_params_}
  - Test Accuracy: {accuracy_score(y_test, y_pred_xgb_labels):.3f}

### Workload Balancing:
- Implemented rule-based workload balancer
- ML-based optimization for batch task assignment
- Capacity utilization tracking

### Key Features:
- Combined TF-IDF + engineered features
- GridSearchCV hyperparameter optimization
- Feature importance analysis
- Workload visualization

## Files Added:
- models/random_forest_priority_optimized.pkl
- models/xgboost_priority_optimized.pkl
- models/workload_balancer.pkl
- results/week3_model_comparison.csv
- results/feature_importance.csv
"""

with open('README_week3_update.md', 'w') as f:
    f.write(readme_update)

print("Week 3 completed successfully!")
print("\nNext: Week 4 - Finalize models and create dashboard")
