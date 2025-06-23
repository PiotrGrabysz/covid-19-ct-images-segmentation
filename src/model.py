import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule

from src import metrics


class UNet(LightningModule):
    def __init__(
        self,
        loss_fn,
        model_name: str = "efficientnet-b0",
        lr=0.001,
        encoder_depth: int = 5,
        metric_to_monitor: str = "val_f1_glass_and_consolidation",
    ):
        super().__init__()
        self.save_hyperparameters()

        # self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            encoder_depth=encoder_depth,
            in_channels=3,
            classes=4,
            activation=None,
        )

        self.loss_fn = loss_fn
        self.lr = lr

        self.preconv = torch.nn.Conv2d(1, 3, kernel_size=1)

        self.metric_to_monitor = metric_to_monitor

    def forward(self, image):
        x = self.preconv(image)
        mask = self.model(x)
        return mask

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def shared_step(self, batch, stage="train"):
        images, true_masks = batch
        logits = self(images)  # (N, 4, H, W)
        loss = self.loss_fn(logits, true_masks)

        probs = torch.sigmoid(logits)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log_metrics(probs, stage, true_masks)
        return loss

    def log_metrics(self, probs, stage, true_masks):
        self.log(
            f"{stage}_f1_glass", metrics.f1score_glass(true_masks, probs), prog_bar=True
        )
        self.log(
            f"{stage}_f1_consolidation",
            metrics.f1score_consolidation(true_masks, probs),
            prog_bar=True,
        )
        self.log(
            f"{stage}_f1_lungs_background",
            metrics.f1score_lungs_other(true_masks, probs),
            prog_bar=True,
        )
        self.log(
            f"{stage}_f1_glass_and_consolidation",
            metrics.f1score_glass_and_consolidation(true_masks, probs),
            prog_bar=True,
        )
        self.log(
            f"{stage}_precision_glass",
            metrics.calculate_precision(true_masks, probs, "glass"),
            prog_bar=True,
        )
        self.log(
            f"{stage}_recall_glass",
            metrics.calculate_recall(true_masks, probs, "glass"),
            prog_bar=True,
        )
        self.log(
            f"{stage}_precision_consolidation",
            metrics.calculate_precision(true_masks, probs, "consolidation"),
            prog_bar=True,
        )
        self.log(
            f"{stage}_recall_consolidation",
            metrics.calculate_recall(true_masks, probs, "consolidation"),
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # since we want to maximize F1
                factor=0.5,
                patience=3,
                verbose=True,
            ),
            "monitor": self.metric_to_monitor,  # must match log name
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
