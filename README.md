# Fake News Detection System

A web-based **Fake News Detection** system using NLP and machine learning. It combines **Multinomial Naive Bayes**, an **LSTM** deep learning model, and a **fine-tuned BERT** transformer, with an **ensemble** (majority + weighted probability voting) for robust predictions.

## Features

- **Three models**: Naive Bayes (TF-IDF), LSTM, fine-tuned BERT
- **Ensemble**: Weighted probability fusion and majority voting
- **Evaluation**: Accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- **Web UI (Streamlit)**:
  - Input news text → **Analyze News** → Final prediction (Fake/Real) + confidence
  - Individual model predictions for transparency
  - Probability bar graph, processing time, Clear/Reset
  - Optional **word highlighting** (important words for the prediction)

## Setup

### 1. Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Dataset

Use the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):

- Download `Fake.csv` and `True.csv`
- Place them in the `data/` folder:

```
FAKE_NEWS_DETECTION_MLPROJ/
  data/
    Fake.csv
    True.csv
```

Alternatively, use a single CSV `data/news_merged.csv` with columns: `title`, `text`, `label` (0 = real, 1 = fake).

### 3. Train models

From the project root:

```bash
python train_all.py
```

For a quick run (small subset, fewer epochs):

```bash
python train_all.py --quick
```

Trained artifacts are saved in `models/` and `artifacts/`.

### 4. Run the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (e.g. http://localhost:8501).

## Project structure

```
├── app.py              # Streamlit application
├── config.py           # Paths and hyperparameters
├── data_loader.py      # Load and split Kaggle data
├── preprocess.py       # Text cleaning and tokenization
├── evaluate.py         # Metrics and ensemble fusion
├── train_all.py        # Train NB, LSTM, BERT and evaluate
├── data/               # Fake.csv, True.csv (or news_merged.csv)
├── models/             # Saved NB, LSTM, BERT
├── artifacts/          # Tokenizers, vectorizer, configs
└── requirements.txt
```

## Configuration

Edit `config.py` to change:

- `ENSEMBLE_WEIGHTS`: weights for [Naive Bayes, LSTM, BERT] (default `[0.2, 0.3, 0.5]`)
- `MAX_SEQUENCE_LENGTH`, LSTM/embedding sizes, BERT model name, train/val/test split

## License

For educational use. Dataset is from Kaggle; check its license for redistribution.
