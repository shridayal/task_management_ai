# 4. Feature Relationships
# Assignee performance analysis
assignee_performance = pd.crosstab(df['Assignee'], df['Status'])
assignee_performance['completion_rate'] = assignee_performance['Done'] / assignee_performance.sum(axis=1)

print("\nAssignee Completion Rates:")
print(assignee_performance['completion_rate'].sort_values(ascending=False))

# Priority completion analysis
priority_completion = pd.crosstab(df['Priority'], df['Status'])
priority_completion['completion_rate'] = priority_completion['Done'] / priority_completion.sum(axis=1)

print("\nPriority Completion Rates:")
print(priority_completion['completion_rate'])

# 5. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 6. Save preprocessed data
df.to_csv('preprocessed_tasks.csv', index=False)
print("\nPreprocessed data saved to 'preprocessed_tasks.csv'")




import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Load your preprocessed data
df = pd.read_csv('preprocessed_tasks.csv')

# 1. TF-IDF Feature Extraction
print("=== TF-IDF Feature Extraction ===")
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_summary'])

print(f"TF-IDF Feature Shape: {tfidf_features.shape}")
print(f"Sample feature names: {tfidf_vectorizer.get_feature_names_out()[:10]}")

# 2. Word2Vec Embeddings (Alternative approach)
print("\n=== Word2Vec Feature Extraction ===")
# Tokenize for Word2Vec
df['tokens'] = df['processed_summary'].apply(lambda x: x.split())

# Train Word2Vec model
w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Create document embeddings by averaging word vectors
def get_doc_embedding(tokens, model):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

w2v_features = np.array([get_doc_embedding(tokens, w2v_model) for tokens in df['tokens']])
print(f"Word2Vec Feature Shape: {w2v_features.shape}")






# Prepare target variables for different classification tasks
# Let's classify based on 'Label' (task type) and 'Priority'

# Task 1: Label Classification
print("\n=== TASK CLASSIFICATION BY LABEL ===")
X_tfidf = tfidf_features
y_label = df['Label']

# Split data
X_train_tfidf, X_test_tfidf, y_train_label, y_test_label = train_test_split(
    X_tfidf, y_label, test_size=0.2, random_state=42, stratify=y_label
)

# Naive Bayes Classifier
print("\n--- Naive Bayes Classifier ---")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train_label)
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Evaluate Naive Bayes
print("Naive Bayes Performance:")
print(f"Accuracy: {accuracy_score(y_test_label, y_pred_nb):.3f}")
print(f"Precision: {precision_score(y_test_label, y_pred_nb, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test_label, y_pred_nb, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_test_label, y_pred_nb, average='weighted'):.3f}")

# SVM Classifier
print("\n--- SVM Classifier ---")
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_tfidf, y_train_label)
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Evaluate SVM
print("SVM Performance:")
print(f"Accuracy: {accuracy_score(y_test_label, y_pred_svm):.3f}")
print(f"Precision: {precision_score(y_test_label, y_pred_svm, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test_label, y_pred_svm, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_test_label, y_pred_svm, average='weighted'):.3f}")



# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test_label, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Naive Bayes Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test_label, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Detailed Classification Report
print("\n=== Detailed Classification Report (SVM) ===")
print(classification_report(y_test_label, y_pred_svm))

# Task 2: Priority Classification
print("\n=== PRIORITY CLASSIFICATION ===")
y_priority = df['Priority']

# Split data
X_train_tfidf_p, X_test_tfidf_p, y_train_priority, y_test_priority = train_test_split(
    X_tfidf, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)

# Train both models for priority
nb_priority = MultinomialNB()
nb_priority.fit(X_train_tfidf_p, y_train_priority)

svm_priority = SVC(kernel='rbf', random_state=42)
svm_priority.fit(X_train_tfidf_p, y_train_priority)

# Predictions
y_pred_nb_priority = nb_priority.predict(X_test_tfidf_p)
y_pred_svm_priority = svm_priority.predict(X_test_tfidf_p)

# Compare models
models_comparison = pd.DataFrame({
    'Model': ['Naive Bayes', 'SVM'],
    'Task_Classification_Accuracy': [
        accuracy_score(y_test_label, y_pred_nb),
        accuracy_score(y_test_label, y_pred_svm)
    ],
    'Priority_Classification_Accuracy': [
        accuracy_score(y_test_priority, y_pred_nb_priority),
        accuracy_score(y_test_priority, y_pred_svm_priority)
    ]
})

print("\n=== Model Comparison ===")
print(models_comparison)
