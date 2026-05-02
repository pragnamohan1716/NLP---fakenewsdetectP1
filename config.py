"""Configuration for Fake News Detection system."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Dataset (Kaggle Fake and Real News)
# Place CSV files in data/ or set path to your download
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
REAL_CSV = os.path.join(DATA_DIR, "True.csv")
# Alternative: single merged file
MERGED_CSV = os.path.join(DATA_DIR, "news_merged.csv")

# --- Column names ---
# Your training artifacts were exported with a label encoding where:
#   - label index for Fake = `LABEL_INDEX_FAKE`
#   - label index for Real = `LABEL_INDEX_REAL`
# The app must interpret model probabilities using this exact mapping.
TITLE_COL = "title"
TEXT_COL = "text"
LABEL_COL = "label"  # numeric label column from the dataset

# Colab/export mapping (fixes “opposite predictions”)
LABEL_INDEX_FAKE = 0
LABEL_INDEX_REAL = 1

# --- Model artifacts (Updated to match your actual files) ---
# Naive Bayes
NB_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes.joblib")
NB_MODEL_COLAB_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")

# LSTM
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.h5") 
LSTM_TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "lstm_tokenizer.joblib")
LSTM_MODEL_COLAB_PATH = os.path.join(MODELS_DIR, "fake_news_lstm.h5")
LSTM_TOKENIZER_COLAB_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")

# BERT
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_fake_news")
BERT_TOKENIZER_PATH = os.path.join(MODELS_DIR, "bert_fake_news")

# Other paths (can remain for compatibility)
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab_and_vectorizer.joblib")
LSTM_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "lstm_config.joblib")
# Training defaults
MAX_SEQUENCE_LENGTH = 200
LSTM_EMBEDDING_DIM = 128
LSTM_UNITS = 64
BERT_MODEL_NAME = "bert-base-uncased"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Ensemble weights (can be tuned from validation performance)
# Order: [Naive Bayes, LSTM, BERT]
ENSEMBLE_WEIGHTS = [0.2, 0.2, 0.6]

# Ensure directories exist
for d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    os.makedirs(d, exist_ok=True)
