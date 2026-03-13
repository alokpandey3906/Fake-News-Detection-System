# Fake News Detection System

This is a machine learning-based fake news detection system built with Python, scikit-learn, and NLTK.

## Features

- Trains a logistic regression model on news articles
- Uses TF-IDF vectorization for text features
- Includes text preprocessing (lowercasing, punctuation removal, stopword removal, lemmatization)
- Evaluates model performance with classification report and confusion matrix
- Provides a prediction script for new articles
- **Web-based UI using Streamlit for easy interaction**

## Requirements

- Python 3.7+
- Libraries: pandas, scikit-learn, nltk, matplotlib, seaborn, streamlit

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Place your datasets in the `Datasets/` folder:
   - `Fake.csv`: CSV with fake news articles (columns: title, text, subject, date)
   - `True.csv`: CSV with real news articles (columns: title, text, subject, date)

2. Run the training script:
```
python main.py
```

This will:
- Load and preprocess the data
- Train the model
- Save the model as `fake_news_model.pkl` and vectorizer as `vectorizer.pkl`
- Generate a confusion matrix plot as `confusion_matrix.png`

### Making Predictions

#### Command Line
Run the prediction script:
```
python predict.py
```

Enter a news title when prompted. The script will output the prediction (Fake or True) along with probabilities.

#### Web UI
For a user-friendly interface, run the Streamlit app:
```
streamlit run app.py
```

This will open a web browser with a simple form to input news title, and display the prediction with confidence scores.

## File Structure

- `main.py`: Training script
- `predict.py`: Command-line prediction script
- `app.py`: Streamlit web app for predictions
- `requirements.txt`: Python dependencies
- `Datasets/`: Folder for training data
- `fake_news_model.pkl`: Trained model (generated)
- `vectorizer.pkl`: TF-IDF vectorizer (generated)
- `confusion_matrix.png`: Evaluation plot (generated)

## Notes

- The system assumes the CSV files have the specified columns.
- For large datasets, consider increasing `max_features` in TfidfVectorizer or using dimensionality reduction.
- Model performance depends on the quality and quantity of training data.

## Deployment

### ✅ Deploy on Streamlit Cloud (Recommended)
1. Create a GitHub repo and push your project.
2. Go to https://share.streamlit.io and connect your GitHub account.
3. Select this repository and branch.
4. Set the main file to `app.py` and deploy.

### 🐳 Run with Docker
Build and run the container:
```bash
docker build -t fake-news-detector .
docker run -p 8501:8501 fake-news-detector
```
Then browse to `http://localhost:8501`.

### 🚀 Deploy on Heroku
1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
2. Login and create an app:
```bash
heroku login
heroku create my-fake-news-detector
```
3. Push the repo:
```bash
git push heroku main
```

Heroku uses the included `Procfile` to start the Streamlit web server.
