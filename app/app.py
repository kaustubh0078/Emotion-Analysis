import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

# --- Page Config ---
st.set_page_config(
    page_title="Emotion Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stTextArea textarea {
        height: 150px;
        font-size: 16px;
    }
    .stButton button {
        height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# --- Preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    text = ''.join([i for i in text if i.isascii()])
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned = [i for i in words if i not in stop_words]
    return ' '.join(cleaned)

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        # Get the absolute path to the directory containing this script (app.py)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path to models (siblings directory)
        models_dir = os.path.join(current_dir, '..', 'models')
        
        pipeline_path = os.path.join(models_dir, 'emotion_pipeline.pkl')
        mapping_path = os.path.join(models_dir, 'emotion_mapping.pkl')
        
        pipeline = joblib.load(pipeline_path)
        emotion_map = joblib.load(mapping_path)
        id_to_emotion = {v: k for k, v in emotion_map.items()}
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please ensure 'emotion_pipeline.pkl' and 'emotion_mapping.pkl' are in the 'models' directory.")
        st.error(f"Debug Info: {e}")
        return None, None
    return pipeline, id_to_emotion

pipeline, id_to_emotion = load_artifacts()

# --- Session State for Input ---
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def set_text(text):
    st.session_state.user_input = text

# --- Sidebar ---
with st.sidebar:
    st.title("Emotion Analyzer")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses Natural Language Processing (NLP) to "
        "detect emotions in text. It supports 6 core emotions: "
        "\n\n1. Joy 😃\n2. Sadness 😔\n3. Anger 😠\n4. Fear 😨\n5. Love ❤️\n6. Surprise 😲"
    )
    st.markdown("---")
    st.write("Created with ❤️ using [Streamlit](https://streamlit.io)")

# --- Main Content ---
st.markdown('<p class="big-font">🎭 Emotion Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("Analyze the underlying emotion of any English sentence in real-time.")

col1, col2 = st.columns([1.5, 1], gap="medium")

with col1:
    st.markdown("### 📝 Enter Text")
    
    # "Try it out" buttons
    st.markdown("Try an example:")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    btn_col1.button("I passed my exam! 🎓", on_click=set_text, args=("I passed my exam! I am so happy and proud.",))
    btn_col2.button("I lost my keys... 🔑", on_click=set_text, args=("I lost my keys and I feel terrible about it.",))
    btn_col3.button("This is annoying 😤", on_click=set_text, args=("This situation is making me very angry and frustrated.",))
    
    # Input Area
    text_input = st.text_area(
        "", 
        key="user_input",
        placeholder="Type here to analyze emotion... (e.g., 'I successfully deployed my model today!')",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Prediction Results")
    
    if analyze_btn:
        if st.session_state.user_input.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.5) # UX delay
                
                processed_text = preprocess_text(st.session_state.user_input)
                
                if pipeline:
                    # Get prediction and probabilities
                    prediction = pipeline.predict([processed_text])[0]
                    proba_dist = pipeline.predict_proba([processed_text])[0]
                    max_prob = proba_dist.max()
                    
                    predicted_emotion = id_to_emotion.get(prediction, "Unknown")
                    
                    # Color Mapping
                    emotion_colors = {
                        "joy": "#FFD700",       # Gold
                        "love": "#FF69B4",      # HotPink
                        "anger": "#FF4500",     # OrangeRed
                        "fear": "#8A2BE2",      # BlueViolet
                        "surprise": "#1E90FF",  # DodgerBlue
                        "sadness": "#708090"    # SlateGray
                    }
                    color = emotion_colors.get(predicted_emotion, "black")
                    
                    # Display Main Result
                    st.markdown(f"**Detected Emotion:**")
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text_align: center;">
                            <h1 style="color: white; margin: 0; text-align: center;">{predicted_emotion.upper()}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Confidence Score:**")
                    st.progress(max_prob)
                    st.caption(f"{max_prob:.1%} certainty")
                    
                    st.divider()
                    
                    # Detailed Probs
                    with st.expander("See full probability distribution"):
                        # Sort by probability descending
                        probs_list = [(id_to_emotion.get(i), p) for i, p in enumerate(proba_dist)]
                        probs_list.sort(key=lambda x: x[1], reverse=True)
                        
                        for emo_name, prob in probs_list:
                            st.write(f"**{emo_name.title()}**: {prob:.1%}")
                            st.progress(prob)
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    else:
        st.info("👈 Enter text and click 'Analyze Emotion' to see results here.")
