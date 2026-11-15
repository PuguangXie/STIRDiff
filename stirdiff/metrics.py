"""Small, dependency-light metrics used for model selection and reporting."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _validated_arrays(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("labels and probabilities must be nonempty vectors of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y.astype(np.int8, copy=False), p


def youden_threshold(labels, probabilities) -> float:
    """Choose a binary classification cutoff by maximizing Youden's J."""
    y, p = _validated_arrays(labels, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("Youden threshold requires both label classes")
    fpr, tpr, thresholds = roc_curve(y, p)
    usable = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if not usable.any():
        raise ValueError("No finite classification threshold is available")
    indices = np.flatnonzero(usable)
    return float(thresholds[indices[np.argmax((tpr - fpr)[usable])]])


def basic_metrics(labels, probabilities, threshold=None) -> dict:
    """Return AUROC, average precision and Brier score, plus optional class metrics.

    Undefined AUROC/AP (single-class labels) are represented by ``None`` so the
    result can be written directly as JSON. Classification metrics are included
    only when a finite threshold in [0, 1] is supplied.
    """
    y, p = _validated_arrays(labels, probabilities)
    both = len(np.unique(y)) == 2
    result = {
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "brier": float(np.mean((p - y) ** 2)),
    }
    if threshold is None:
        return result
    if isinstance(threshold, (bool, np.bool_)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite and in [0, 1]")
    cutoff = float(threshold)
    if not 0 <= cutoff <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    predicted = p >= cutoff
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    tn = int(np.sum(~predicted & negative))
    fn = int(np.sum(~predicted & positive))
    result.update({
        "threshold": cutoff,
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ppv": float(tp / (tp + fp)) if tp + fp else None,
        "npv": float(tn / (tn + fn)) if tn + fn else None,
    })
    return result


__all__ = ["basic_metrics", "youden_threshold"]
