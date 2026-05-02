"""Evaluation metrics and ensemble fusion (majority + weighted voting)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_proba=None):
    """Compute accuracy, precision, recall, F1, confusion matrix, and optionally ROC-AUC."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def ensemble_majority_vote(preds_list):
    """
    preds_list: list of (N,) arrays of binary predictions (0 or 1).
    Returns (N,) array of majority vote.
    """
    stacked = np.stack(preds_list, axis=1)
    return (stacked.sum(axis=1) > (stacked.shape[1] / 2)).astype(int)


def ensemble_weighted_probability(proba_list, weights=None):
    """
    probs_list: list of (N, 2) arrays [P(real), P(fake)].
    weights: optional list of weights (same length as proba_list). Default: equal.
    Returns (N, 2) averaged probabilities and (N,) predicted class.
    """
    if weights is None:
        weights = np.ones(len(proba_list)) / len(proba_list)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    avg_proba = sum(w * p for w, p in zip(weights, proba_list))
    preds = (avg_proba[:, 1] >= 0.5).astype(int)
    return avg_proba, preds


def fuse_predictions(proba_list, weights=None, use_majority=True):
    """
    Combine predictions: weighted probability average for final score,
    optionally also compute majority vote and return both.
    Returns dict with:
      - weighted_proba: (N, 2)
      - weighted_pred: (N,)
      - majority_pred: (N,) if use_majority else None
    """
    weighted_proba, weighted_pred = ensemble_weighted_probability(proba_list, weights)
    out = {"weighted_proba": weighted_proba, "weighted_pred": weighted_pred}
    if use_majority:
        preds_list = [(p[:, 1] >= 0.5).astype(int) for p in proba_list]
        out["majority_pred"] = ensemble_majority_vote(preds_list)
    else:
        out["majority_pred"] = None
    return out
