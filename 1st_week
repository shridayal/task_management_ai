import pandas as pd
import numpy as np
df = pd.read_csv('fake_jira_dataset.csv')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Data Type Conversions and Basic Cleaning
df['Created'] = pd.to_datetime(df['Created'])
df['Year'] = df['Created'].dt.year
df['Month'] = df['Created'].dt.month

# 2. Distribution Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Status distribution
df['Status'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Task Status Distribution')
axes[0,0].set_xlabel('Status')

# Priority distribution
df['Priority'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightcoral')
axes[0,1].set_title('Task Priority Distribution')
axes[0,1].set_xlabel('Priority')

# Label distribution
df['Label'].value_counts().plot(kind='bar', ax=axes[0,2], color='lightgreen')
axes[0,2].set_title('Task Label Distribution')
axes[0,2].set_xlabel('Label')

# Assignee workload
df['Assignee'].value_counts().plot(kind='bar', ax=axes[1,0], color='gold')
axes[1,0].set_title('Tasks per Assignee')
axes[1,0].set_xlabel('Assignee')

# Tasks over time
df.groupby(df['Created'].dt.to_period('M')).size().plot(ax=axes[1,1], color='purple')
axes[1,1].set_title('Tasks Created Over Time')
axes[1,1].set_xlabel('Month')

# Priority vs Status heatmap
priority_status = pd.crosstab(df['Priority'], df['Status'])
sns.heatmap(priority_status, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,2])
axes[1,2].set_title('Priority vs Status Heatmap')

plt.tight_layout()
plt.show()

# 3. Text Analysis of Summary Field
df['summary_length'] = df['Summary'].str.len()
df['word_count'] = df['Summary'].str.split().str.len()





import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocess task summary text for NLP analysis
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['processed_summary'] = df['Summary'].apply(preprocess_text)

# Display sample
print("\nOriginal vs Processed Summary Examples:")
for i in range(3):
    print(f"\nOriginal: {df['Summary'].iloc[i]}")
    print(f"Processed: {df['processed_summary'].iloc[i]}")

print("\nSummary Text Statistics:")
print(f"Average summary length: {df['summary_length'].mean():.2f} characters")
print(f"Average word count: {df['word_count'].mean():.2f} words")
