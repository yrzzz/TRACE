#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auc_metrics(y_true, y_proba, classes, pos_label: str = "Fibroblasts"):
    num_classes = len(classes)
    metrics = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "pos_idx": None,
        "error": None,
    }

    try:
        if num_classes == 2:
            if pos_label in classes:
                pos_idx = int(np.where(classes == pos_label)[0][0])
            else:
                pos_idx = 1
            y_bin = (y_true == pos_idx).astype(int)
            metrics["roc_auc"] = roc_auc_score(y_bin, y_proba[:, pos_idx])
            metrics["pr_auc"] = average_precision_score(y_bin, y_proba[:, pos_idx])
            metrics["pos_idx"] = pos_idx
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
    except Exception as e:
        metrics["error"] = str(e)

    return metrics
