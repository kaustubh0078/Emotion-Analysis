import pandas as pd
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Preprocessing Functions ---
# We will create a custom Transformer for preprocessing to include it in the pipeline
# This ensures that raw text going into the pipeline (during inference) gets processed exactly like training data

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._preprocess)

    def _preprocess(self, text):
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 3. Remove Numbers
        text = ''.join([i for i in text if not i.isdigit()])
        
        # 4. Remove Emojis (keep only ascii)
        text = ''.join([i for i in text if i.isascii()])
        
        # 5. Tokenize and Remove Stopwords
        words = word_tokenize(text)
        cleaned = [i for i in words if i not in self.stop_words]
        
        return ' '.join(cleaned)

def main():
    print("Loading dataset...")
    try:
        # Data is now in ../data/train.txt relative to src/
        df = pd.read_csv('../data/train.txt', sep=';', header=None, names=['text', 'emotion'])
    except FileNotFoundError:
        print("Error: 'train.txt' not found. Please ensure the dataset is in the '../data/' directory.")
        return

    # Map emotions to numbers (as done in notebook)
    unique_emotions = df['emotion'].unique()
    emotion_numbers = {emo: i for i, emo in enumerate(unique_emotions)}
    df['emotion_num'] = df['emotion'].map(emotion_numbers)
    
    # Save the mapping
    joblib.dump(emotion_numbers, '../models/emotion_mapping.pkl')
    print(f"Emotion mapping saved to '../models/emotion_mapping.pkl': {emotion_numbers}")

    print("Training model pipeline...")
    
    # ... (rest of logic same) ...
    
    # Let's clean the data first for training, as per notebook logic
    preprocessor = TextPreprocessor()
    X = preprocessor.transform(df['text'])
    y = df['emotion_num']

    # Vectorizer + Model Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    print("Model trained successfully.")

    # Save the pipeline
    joblib.dump(pipeline, '../models/emotion_pipeline.pkl')
    print("Pipeline saved as '../models/emotion_pipeline.pkl'")

if __name__ == "__main__":
    main()
