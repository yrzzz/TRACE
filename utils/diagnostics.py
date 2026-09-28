from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)

def compute_metrics_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_strategy: str = "fixed",
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thr = float(threshold)
    if threshold_strategy == "best_f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        if thresholds.size > 0:
            f1 = _safe_div(2 * precision[1:] * recall[1:], precision[1:] + recall[1:])
            best_idx = int(np.nanargmax(f1))
            thr = float(thresholds[best_idx])

    y_pred = (y_score >= thr).astype(int)
    return {
        "threshold": thr,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def compute_metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
