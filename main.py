import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
print("Loading datasets...")
fake_df = pd.read_csv('Datasets/Fake.csv')
true_df = pd.read_csv('Datasets/True.csv')

# Add labels
fake_df['label'] = 1  # Fake
true_df['label'] = 0  # True

# Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution: {df['label'].value_counts()}")
print(df.head())

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize (simple split)
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove short words
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to title + text
print("Preprocessing text...")
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['combined_text'] = df['title'] + ' ' + df['text']
df['processed_text'] = df['combined_text'].apply(preprocess_text)

# Split data
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Vectorize
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Add bigrams
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
# model = LogisticRegression(random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
print("Evaluating...")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
# plt.show()  # Remove to avoid hanging in terminal

# Save model and vectorizer
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved as fake_news_model.pkl and vectorizer.pkl")