from typing import Literal

from torchmetrics.functional import f1_score, precision, recall

class_to_idx = {"glass": 0, "consolidation": 1}


def f1score_glass(y_true, y_pred):
    return f1_score(
        y_pred[:, 0:1, ...],
        y_true[:, 0:1, ...],
        task="binary",
        threshold=0.5,
    )


def f1score_consolidation(y_true, y_pred):
    return f1_score(
        y_pred[:, 1:2, ...],
        y_true[:, 1:2, ...],
        task="binary",
        threshold=0.5,
    )


def f1score_glass_and_consolidation(y_true, y_pred):
    return f1_score(
        y_pred[:, [0, 1], :, :],
        y_true[:, [0, 1], :, :],
        task="multilabel",
        num_labels=2,
        threshold=0.5,
        average="macro",
    )


def f1score_lungs_other(y_true, y_pred):
    return f1_score(
        y_pred[:, [2, 3], :, :],
        y_true[:, [2, 3], :, :],
        task="multilabel",
        num_labels=2,
        threshold=0.5,
        average="macro",
    )


def calculate_precision(y_true, y_pred, class_name: Literal["glass", "consolidation"]):
    class_idx = class_to_idx[class_name]
    return precision(
        y_pred[:, class_idx, ...], y_true[:, class_idx, ...], task="binary"
    )


def calculate_recall(y_true, y_pred, class_name: Literal["glass", "consolidation"]):
    class_idx = class_to_idx[class_name]
    return recall(y_pred[:, class_idx, ...], y_true[:, class_idx, ...], task="binary")
