"""
Train all three models and evaluate.
Run once to produce saved models; the Streamlit app will load them for inference.
Usage: python train_all.py
Requires: data/Fake.csv and data/True.csv (or data/news_merged.csv) in project root.
"""
import sys
import numpy as np

import config
from data_loader import load_kaggle_data, combine_title_text, get_train_val_test
from models.naive_bayes_model import build_pipeline, train_nb, predict_proba_nb
from models.lstm_model import train_lstm, load_lstm, predict_proba_lstm
from models.bert_model import train_bert, load_bert, predict_proba_bert
from evaluate import evaluate_predictions, fuse_predictions

# Optional: reduce BERT/LSTM scope for quick run
QUICK_RUN = "--quick" in sys.argv  # smaller data and fewer epochs


def main():
    print("Loading dataset...")
    df = load_kaggle_data()
    df = combine_title_text(df)
    train, val, test = get_train_val_test(df)

    if QUICK_RUN:
        train = train.sample(n=min(2000, len(train)), random_state=config.RANDOM_STATE)
        val = val.sample(n=min(400, len(val)), random_state=config.RANDOM_STATE)
        test = test.sample(n=min(500, len(test)), random_state=config.RANDOM_STATE)
        print("Quick run: using subset of data.")

    X_train = train["content"].tolist()
    y_train = train[config.LABEL_COL].values
    X_val = val["content"].tolist()
    y_val = val[config.LABEL_COL].values
    X_test = test["content"].tolist()
    y_test = test[config.LABEL_COL].values

    # --- Naive Bayes ---
    print("Training Multinomial Naive Bayes...")
    pipeline = train_nb(X_train, y_train)
    nb_proba_test = predict_proba_nb(pipeline, X_test)
    nb_pred = (nb_proba_test[:, 1] >= 0.5).astype(int)
    nb_metrics = evaluate_predictions(y_test, nb_pred, nb_proba_test)
    print("NB - Accuracy: {:.4f}, F1: {:.4f}, ROC-AUC: {:.4f}".format(
        nb_metrics["accuracy"], nb_metrics["f1"], nb_metrics["roc_auc"]))

    # --- LSTM ---
    print("Training LSTM...")
    train_lstm(
        X_train, y_train, X_val, y_val,
        epochs=3 if QUICK_RUN else 5,
        batch_size=64,
    )
    lstm_model, lstm_tok, maxlen = load_lstm()
    lstm_proba_test = predict_proba_lstm(lstm_model, lstm_tok, maxlen, X_test)
    lstm_pred = (lstm_proba_test[:, 1] >= 0.5).astype(int)
    lstm_metrics = evaluate_predictions(y_test, lstm_pred, lstm_proba_test)
    print("LSTM - Accuracy: {:.4f}, F1: {:.4f}, ROC-AUC: {:.4f}".format(
        lstm_metrics["accuracy"], lstm_metrics["f1"], lstm_metrics["roc_auc"]))

    # --- BERT ---
    print("Training BERT (this may take a while)...")
    train_bert(
        X_train, y_train, X_val, y_val,
        epochs=2 if QUICK_RUN else 3,
        batch_size=16,
        max_length=128,
    )
    bert_model, bert_tok = load_bert()
    bert_proba_test = predict_proba_bert(bert_model, bert_tok, X_test, max_length=128)
    bert_pred = (bert_proba_test[:, 1] >= 0.5).astype(int)
    bert_metrics = evaluate_predictions(y_test, bert_pred, bert_proba_test)
    print("BERT - Accuracy: {:.4f}, F1: {:.4f}, ROC-AUC: {:.4f}".format(
        bert_metrics["accuracy"], bert_metrics["f1"], bert_metrics["roc_auc"]))

    # --- Ensemble ---
    fused = fuse_predictions(
        [nb_proba_test, lstm_proba_test, bert_proba_test],
        weights=config.ENSEMBLE_WEIGHTS,
        use_majority=True,
    )
    ens_pred = fused["weighted_pred"]
    ens_metrics = evaluate_predictions(y_test, ens_pred, fused["weighted_proba"])
    print("Ensemble (weighted) - Accuracy: {:.4f}, F1: {:.4f}, ROC-AUC: {:.4f}".format(
        ens_metrics["accuracy"], ens_metrics["f1"], ens_metrics["roc_auc"]))

    print("\nConfusion matrix (Ensemble):")
    print(ens_metrics["confusion_matrix"])
    print("\nTraining complete. Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
