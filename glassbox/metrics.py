"""Evaluation metrics: predictive quality, fidelity and interpretability."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def classification_metrics(y_true, prob, threshold=config.DECISION_THRESHOLD) -> dict:
    """Accuracy, precision, recall, F1 and ROC-AUC against the ground truth."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def fidelity(surrogate_labels, nn_labels) -> float:
    """Fraction of inputs on which the surrogate makes the same decision as the network."""
    return float(np.mean(np.asarray(surrogate_labels) == np.asarray(nn_labels)))


def fidelity_by_confidence(surrogate_labels, nn_labels, nn_prob, band=0.15) -> dict:
    """Fidelity split into points where the network is unsure vs. confident.

    "Boundary" points have |P(diabetic) - 0.5| < band. Surrogates typically
    struggle there, because that is where the network's decision surface bends.
    """
    surrogate_labels = np.asarray(surrogate_labels)
    nn_labels = np.asarray(nn_labels)
    near = np.abs(np.asarray(nn_prob) - 0.5) < band
    out = {"n_boundary": int(near.sum()), "n_confident": int((~near).sum())}
    out["boundary"] = fidelity(surrogate_labels[near], nn_labels[near]) if near.any() else float("nan")
    out["confident"] = fidelity(surrogate_labels[~near], nn_labels[~near]) if (~near).any() else float("nan")
    return out


def probability_mae(surrogate_prob, nn_prob) -> float:
    """Mean absolute gap between surrogate output and network probability."""
    return float(np.mean(np.abs(np.asarray(surrogate_prob) - np.asarray(nn_prob))))


def anfis_complexity(model) -> dict:
    """Interpretability measures of a fuzzy rule base (smaller = easier to read)."""
    A = model.antecedents_
    used = A >= 0
    return {
        "rules": int(A.shape[0]),
        "conditions": int(used.sum()),
        "avg_rule_length": float(used.sum(axis=1).mean()),
        "features_used": int(used.any(axis=0).sum()),
        "terms_per_feature": int(model.n_terms),
        "then_parameters": int(np.size(model.consequents_)),   # numbers a reader must take in after THEN
    }
