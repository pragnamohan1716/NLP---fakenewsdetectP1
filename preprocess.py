"""Text preprocessing shared across models (e.g. for explainability and NB)."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data on first run
try:
    stopwords.words("english")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_for_display(text):
    """Clean text for display (same as data_loader.clean_text)."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_importance(text):
    """Tokenize after cleaning for word-level importance (e.g. NB coefficients)."""
    cleaned = clean_for_display(text)
    tokens = word_tokenize(cleaned)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
