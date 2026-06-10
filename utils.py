"""
Metrics and checkpoint utilities.
"""

from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true, y_pred, y_scores=None, average: str = "binary"
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics["roc_auc"] = auc
        except Exception:
            pass

    return metrics


def save_checkpoint(model: nn.Module, path: str):
    state = {"model_state_dict": model.state_dict()}
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    return model