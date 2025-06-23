import segmentation_models_pytorch as smp


def build_loss(alpha: float = 0.5):
    if alpha < 0 or alpha > 1:
        return ValueError("Parameter alpha must be in range [0, 1]")

    bce = smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

    def hybrid_loss(logits, targets):
        return alpha * bce(logits, targets) + (1 - alpha) * dice(logits, targets)

    return hybrid_loss
