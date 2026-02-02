# Emotion Analysis App

A Streamlit application that detects emotions from text using a Logistic Regression model trained on emotion datasets.

## 📂 Project Structure

- `app/`: Contains the Streamlit application code (`app.py`).
- `data/`: Contains the training dataset (`train.txt`).
- `models/`: Stores the trained model pipelines (`emotion_pipeline.pkl`) and mappings.
- `notebooks/`: Exploratory Data Analysis (EDA) and prototype modeling (`Emotion_Analysis.ipynb`).
- `src/`: Source code for retraining the model pipeline (`train_pipeline.py`).

## 🚀 Installation

1.  Clone this repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download necessary NLTK data (automatically handled by the scripts, but if you run into issues):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## 🛠 Usage

### Running the App
To start the web interface:

```bash
streamlit run app/app.py
```

Then open your browser at the URL shown (usually `http://localhost:8501`).

### Retraining the Model
If you want to retrain the model (e.g., after adding new data to `data/train.txt`):

```bash
cd src
python train_pipeline.py
```
This will generate new artifacts in the `models/` directory.

## 🤖 Model Details
- **Algorithm**: Logistic Regression with TF-IDF Vectorization.
- **Preprocessing**: Lowercase, punctuation removal, stopword removal, lemmatization (via Pipeline).
- **Emotions**: Sadness, Anger, Love, Surprise, Fear, Joy.

## 📝 License
[MIT](LICENSE)
