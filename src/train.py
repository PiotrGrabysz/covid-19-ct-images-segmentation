import argparse
import os
from pathlib import Path
from typing import Self

import albumentations
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from src import metrics

SOURCE_SIZE = 512
TARGET_SIZE = 256

METRIC_TO_MONITOR = "val_f1_glass_and_consolidation"

checkpoint_callback = ModelCheckpoint(
    monitor=METRIC_TO_MONITOR, mode="max", save_top_k=1, filename="best-model"
)

early_stop_callback = EarlyStopping(
    monitor=METRIC_TO_MONITOR,
    mode="max",
    patience=6,  # stop if no improvement after 5 epochs
    verbose=True,
)
tensorboard_logger = TensorBoardLogger("lightning_logs", name="covid_segmentation")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    # Augmentations
    train_transforms = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ElasticTransform(
                alpha=100,  # Strength of the distortion
                sigma=10,  # Smoothing
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            albumentations.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                translate_percent=(0.05, 0.05),
                shear=(-5, 5),
                p=0.7
            ),
            albumentations.RandomSizedCrop(
                min_max_height=(int(SOURCE_SIZE * 0.75), SOURCE_SIZE),
                size=(TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_NEAREST,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = albumentations.Compose(
        [
            albumentations.Resize(
                TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST
            ),
            ToTensorV2(),
        ]
    )

    train_dataset = NumpyDataset.from_path(args.train, transforms=train_transforms)
    test_dataset = NumpyDataset.from_path(args.test, transforms=test_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = UNet()

    trainer = Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_dataloader, test_dataloader)


class NumpyDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        transforms: albumentations.Compose | None = None,
    ):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    @classmethod
    def from_path(
        cls, data_path: Path, transforms: albumentations.Compose | None = None
    ) -> Self:
        images = np.load(data_path / "images.npy").astype(np.float32)
        masks = np.load(data_path / "masks.npy").astype(np.int8)

        return NumpyDataset(images, masks, transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        mask = torch.tensor(mask.permute(2, 0, 1), dtype=torch.float32)

        return image, mask


class UNet(LightningModule):
    def __init__(self, model_name: str = "efficientnet-b0", lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        # self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=4,
            activation=None,
        )

        self.loss_fn = build_loss(alpha=0.5)
        self.lr = lr

        self.preconv = torch.nn.Conv2d(1, 3, kernel_size=1)

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
        self.log(
            f"{stage}_f1_glass", metrics.fscore_glass(true_masks, probs), prog_bar=True
        )
        self.log(
            f"{stage}_f1_consolidation",
            metrics.fscore_consolidation(true_masks, probs),
            prog_bar=True,
        )
        self.log(
            f"{stage}_f1_lungs_background",
            metrics.fscore_lungs_other(true_masks, probs),
            prog_bar=True,
        )
        self.log(
            f"{stage}_f1_glass_and_consolidation",
            metrics.fscore_glass_and_consolidation(true_masks, probs),
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # since we want to maximize F1
                factor=0.5,
                patience=3,
                verbose=True
            ),
            'monitor': METRIC_TO_MONITOR,  # must match log name
            'interval': 'epoch',
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_loss(alpha: float = 0.5):
    if alpha < 0 or alpha > 1:
        return ValueError("Parameter alpha must be in range [0, 1]")

    bce = smp.losses.SoftBCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

    def hybrid_loss(logits, targets):
        return alpha * bce(logits, targets) + (1 - alpha) * dice(logits, targets)

    return hybrid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training"),
        metavar="TEXT",
        help="folder where the training data is saved",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
        metavar="TEXT",
        help="folder where the test data is saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    args = parser.parse_args()

    main(args)
