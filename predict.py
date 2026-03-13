import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data if needed
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except:
    nltk.download('wordnet')

# Load model and vectorizer
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function (same as in training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to predict
def predict_news(title):
    processed = preprocess_text(title)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return 'Fake' if pred == 1 else 'True', prob

# Example usage
if __name__ == "__main__":
    title = input("Enter news title: ")
    label, probs = predict_news(title)
    print(f"Prediction: {label}")
    print(f"Probability of True: {probs[0]:.2f}, Fake: {probs[1]:.2f}")