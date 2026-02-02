import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (cached)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# --- Preprocessing Function (Must match training!) ---
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove Numbers
    text = ''.join([i for i in text if not i.isdigit()])
    
    # 4. Remove Emojis (keep only ascii)
    text = ''.join([i for i in text if i.isascii()])
    
    # 5. Tokenize and Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned = [i for i in words if i not in stop_words]
    
    return ' '.join(cleaned)

# --- Load Model and Mapping ---
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load('../models/emotion_pipeline.pkl')
        emotion_map = joblib.load('../models/emotion_mapping.pkl')
        # Reverse mapping: number -> emotion name
        # The mapping saved was {'sadness': 0, ...}, we need {0: 'sadness', ...}
        id_to_emotion = {v: k for k, v in emotion_map.items()}
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'emotion_pipeline.pkl' and 'emotion_mapping.pkl' are in the 'models' directory.")
        return None, None
        
    return pipeline, id_to_emotion

pipeline, id_to_emotion = load_artifacts()

# --- Streamlit UI ---
st.title("🎭 Emotion Analysis App")
st.write("Enter a sentence to analyze its underlying emotion.")

user_input = st.text_area("How are you feeling?", placeholder="e.g., I successfully deployed my model today!")

if st.button("Analyze Emotion"):
    if user_input.strip():
        # Preprocess
        processed_text = preprocess_text(user_input)
        
        # Predict
        # Pipeline expects an iterable (like a Series or list), so we wrap in a list
        prediction = pipeline.predict([processed_text])[0]
        probability = pipeline.predict_proba([processed_text]).max()
        
        predicted_emotion = id_to_emotion.get(prediction, "Unknown")
        
        # Display Result
        st.success(f"**Emotion:** {predicted_emotion.upper()}")
        st.info(f"**Confidence:** {probability:.2%}")
    else:
        st.warning("Please enter some text first.")
