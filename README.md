# Task Management AI System

## Description
This project implements an AI-powered task classification, prioritization, and workload balancing system using Natural Language Processing (NLP) and Machine Learning techniques.

## Features
- Task classification using Naive Bayes and SVM
- Priority prediction using Random Forest and XGBoost
- Intelligent workload balancing
- Text preprocessing with NLTK
- Feature extraction using TF-IDF and Word2Vec

## Technologies Used
- Python 3.x
- scikit-learn
- XGBoost
- NLTK
- Pandas
- NumPy
- Matplotlib/Seaborn

## Models Implemented
1. Naive Bayes Classifier
2. Support Vector Machine (SVM)
3. Random Forest (with GridSearchCV optimization)
4. XGBoost (with hyperparameter tuning)
5. Ensemble Model

## Performance
- Best Model: XGBoost
- Accuracy: ~93%
- Features: TF-IDF + Engineered features

## How to Run
```python
# Install dependencies
pip install pandas numpy scikit-learn xgboost nltk matplotlib seaborn

# Run the script
python task_management_ai.py
