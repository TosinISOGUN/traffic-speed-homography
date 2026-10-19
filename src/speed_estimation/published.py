"""The evaluation behind the published Table 1.

The manuscript's evaluation script (matic.py, August 2024) scored two hand-entered
lists of 15 binary labels with scikit-learn's accuracy_score, precision_score,
recall_score and f1_score. Its comment reads "1 indicates correct speed
estimation, 0 indicates incorrect speed estimation". No true speeds were
recorded. The labels are reproduced here verbatim, and the four metrics are
computed with the same (standard binary-classification) definitions, so
scikit-learn is not needed.
"""
from __future__ import annotations

# Verbatim from matic.py. One entry per vehicle (15 vehicles).
GROUND_TRUTH = (1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1)
PREDICTIONS = (1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1)

# Table 1 of the manuscript (two decimals).
PUBLISHED = {"accuracy": 0.80, "precision": 0.90, "recall": 0.82, "f1": 0.86}


def binary_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall and F1 with 1 as the positive class,
    matching scikit-learn's defaults (including true negatives in accuracy)."""
    if len(y_true) != len(y_pred):
        raise ValueError("label lists differ in length")
    pairs = list(zip(y_true, y_pred))
    tp = sum(t == 1 and p == 1 for t, p in pairs)
    fp = sum(t == 0 and p == 1 for t, p in pairs)
    fn = sum(t == 1 and p == 0 for t, p in pairs)
    tn = sum(t == 0 and p == 0 for t, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(pairs), "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(pairs),
    }


def reproduce_table1() -> dict:
    return binary_metrics(GROUND_TRUTH, PREDICTIONS)
