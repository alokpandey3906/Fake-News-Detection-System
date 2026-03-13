import streamlit as st
import pickle
import re
import time
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
@st.cache_resource
def load_model():
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Preprocessing function
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

# Prediction function
def predict_news(title):
    processed = preprocess_text(title)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return 'Fake News' if pred == 1 else 'Real News', prob

# Streamlit UI
st.set_page_config(
    page_title="🕵️ Fake News Detector Pro",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for enchanting design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        animation: fadeInUp 1s ease-out;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        animation: fadeInUp 1.2s ease-out;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .input-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .result-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .analyze-btn {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border: none;
        color: white;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .analyze-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .confidence-bar {
        height: 25px;
        border-radius: 15px;
        margin: 1rem 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stats-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: left;
        color: #333;
    }
    
    .emoji-float {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
    }
    
    .feature-list li {
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

# Header
# Header with floating emojis
st.markdown('<h1 class="main-header">📰 Fake News Detection System <span class="emoji-float">🕵️‍♂️</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze news titles to detect potential fake news using advanced machine learning <span class="emoji-float">🤖</span></p>', unsafe_allow_html=True)

# Sidebar with enhanced content
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("## 🎯 About This Tool")
    st.markdown("This AI-powered system uses machine learning to analyze news titles and detect potential fake news.")
    
    st.markdown("### ✨ Features")
    st.markdown("""
    <ul class="feature-list">
        <li>🔍 Real-time analysis</li>
        <li>📊 Confidence scoring</li>
        <li>🎨 Beautiful interface</li>
        <li>⚡ Fast processing</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Model Stats")
    st.markdown('<div class="stats-card">', unsafe_allow_html=True)
    st.markdown("**Accuracy:** 76%")
    st.markdown("**Training Samples:** 200")
    st.markdown("**Features:** TF-IDF + N-grams")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Enter complete news titles for best results
    - Check confidence levels for uncertainty
    - Use for educational purposes
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content in columns with animations
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Enter News Title")
    with st.container():
        st.markdown('<div class="input-container fade-in">', unsafe_allow_html=True)
        title = st.text_input("News Title", placeholder="Type or paste the news title here...", key="title_input", label_visibility="collapsed")
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    result_placeholder = st.empty()
    with result_placeholder.container():
        st.markdown('<div class="result-container fade-in">', unsafe_allow_html=True)
        st.info("Enter a title and click 'Analyze News' to see results")
        st.markdown('</div>', unsafe_allow_html=True)

# Analysis logic with enhanced feedback
if analyze_button:
    if title.strip():
        with st.spinner("🔄 Analyzing... Please wait"):
            time.sleep(1)  # Add a small delay for better UX
            label, probs = predict_news(title)
        
        # Update results with animations
        with result_placeholder.container():
            st.markdown('<div class="result-container fade-in">', unsafe_allow_html=True)
            
            if label == "Fake News":
                st.error(f"🚨 **Prediction: {label}**")
                st.markdown("⚠️ This title appears to be from fake news. Stay vigilant! 🛡️")
            else:
                st.success(f"✅ **Prediction: {label}**")
                st.markdown("👍 This title appears to be from real news. Great find! 🎉")
            
            st.markdown("### 📈 Confidence Levels")
            
            # Enhanced progress bars with colors
            col_real, col_fake = st.columns(2)
            with col_real:
                st.markdown(f"**Real News:** {probs[0]*100:.1f}%")
                st.progress(probs[0])
            with col_fake:
                st.markdown(f"**Fake News:** {probs[1]*100:.1f}%")
                st.progress(probs[1])
            
            # Add a fun fact or tip
            if probs[0] > 0.7:
                st.markdown("💡 **Tip:** High confidence in real news - this looks trustworthy!")
            elif probs[1] > 0.7:
                st.markdown("💡 **Tip:** High confidence in fake news - verify with multiple sources!")
            else:
                st.markdown("💡 **Tip:** Close call! Consider checking additional context.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a news title to analyze.")

# Footer with enhanced styling
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("**About:** This AI-powered system analyzes news titles using natural language processing and machine learning. Results are for informational purposes only.")
st.markdown("Built with ❤️ using Streamlit and scikit-learn | Made enchanting with custom CSS ✨")
st.markdown('</div>', unsafe_allow_html=True)