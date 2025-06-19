from torchmetrics.functional import f1_score


def fscore_glass(y_true, y_pred):
    return f1_score(
        y_pred[:, 0:1, ...],
        y_true[:, 0:1, ...],
        task="binary",
        threshold=0.5,
    )


def fscore_consolidation(y_true, y_pred):
    return f1_score(
        y_pred[:, 1:2, ...],
        y_true[:, 1:2, ...],
        task="binary",
        threshold=0.5,
    )


def fscore_glass_and_consolidation(y_true, y_pred):
    return f1_score(
        y_pred[:, [0, 1], :, :],
        y_true[:, [0, 1], :, :],
        task="multilabel",
        num_labels=2,
        threshold=0.5,
        average="macro",
    )


def fscore_lungs_other(y_true, y_pred):
    return f1_score(
        y_pred[:, [2, 3], :, :],
        y_true[:, [2, 3], :, :],
        task="multilabel",
        num_labels=2,
        threshold=0.5,
        average="macro",
    )
