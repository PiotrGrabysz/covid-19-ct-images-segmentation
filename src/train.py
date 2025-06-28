import os
from pathlib import Path

import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from typing_extensions import Annotated

from data.augmentation import AugmentationProbability
from data.data_loaders import build_data_loaders
from loss import build_loss
from model import UNet

SOURCE_SIZE = 512
TARGET_SIZE = 256

METRIC_TO_MONITOR = "val_f1_glass_and_consolidation"


def main(
    train: Annotated[
        Path, typer.Option(help="folder where the training data is saved")
    ] = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training"),
    test: Annotated[
        Path, typer.Option(help="folder where the test data is saved")
    ] = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
    train_batch_size: Annotated[
        int, typer.Option(help="input batch size for training")
    ] = 24,
    test_batch_size: Annotated[
        int, typer.Option(help="input batch size for testing")
    ] = 24,
    epoch: Annotated[int, typer.Option(help="Maximum number of epochs to train")] = 5,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.001,
    architecture: Annotated[str, typer.Option(help="the network used to train the segmentation model. You can choose from 'unet', 'unet++', 'fpn', 'pspnet'")] = "unet",
    alpha: Annotated[
        float,
        typer.Option(
            help=(
                "proportion between Binary Cross Entropy and Dice loss. "
                "alpha=1 is equal to BCE only and alpha=0 is equal to Dice only. "
                "The value of alpha must be between 0 and 1."
            )
        ),
    ] = 0.5,
    encoder_depth: Annotated[
        int, typer.Option(help="a number of stages used in encoder in range [3, 5]")
    ] = 5,
    backbone: Annotated[str, "backbone used in the encoder"] = "efficientnet-b0",
    tensorboard_dir: Annotated[
        Path, typer.Option(help="directory where tensorboard saves logs)")
    ] = "/opt/ml/output/tensorboard",
    checkpoint_dir: str | None = os.environ.get("SM_MODEL_DIR"),
    num_workers: Annotated[
        int,
        typer.Option(
            help="how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. "
        ),
    ] = 0,
    dry_run: Annotated[bool, typer.Option(help="quickly check a single pass")] = False,
    horizontal_flip: float = 0.5,
    elastic_transform: float = 0.5,
    affine: float = 0.7,
    brightness: float = 0.5,
    noise: float = 0.5,
    random_sized_crop: int = 1,
    elastic_transform_strength: Annotated[
        float, typer.Option(help="strength of the Elastic Transform augmentation")
    ] = 100,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    augmentation_probability = AugmentationProbability(
        horizontal_flip=horizontal_flip,
        elastic_transform=elastic_transform,
        affine=affine,
        brightness=brightness,
        noise=noise,
        random_sized_crop=random_sized_crop,
    )

    train_dataloader, test_dataloader = build_data_loaders(
        train=train,
        test=test,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        img_source_size=SOURCE_SIZE,
        img_target_size=TARGET_SIZE,
        num_workers=num_workers,
        elastic_transform_strength=elastic_transform_strength,
        augmentation_probability=augmentation_probability,
    )

    loss_fn = build_loss(alpha=alpha)

    model = UNet(
        metric_to_monitor=METRIC_TO_MONITOR,
        lr=lr,
        encoder_depth=encoder_depth,
        loss_fn=loss_fn,
        backbone_name=backbone,
    )

    tensorboard_logger = TensorBoardLogger(tensorboard_dir, name="covid_segmentation")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=METRIC_TO_MONITOR,
        mode="max",
        save_top_k=1,
        filename="best-model",
    )

    early_stop_callback = EarlyStopping(
        monitor=METRIC_TO_MONITOR,
        mode="max",
        patience=6,  # stop if no improvement after 5 epochs
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        log_every_n_steps=5,
        fast_dev_run=dry_run,
    )
    trainer.fit(model, train_dataloader, test_dataloader)

    best_metric = checkpoint_callback.best_model_score
    print(f"Best {METRIC_TO_MONITOR}: {best_metric}")


if __name__ == "__main__":
    typer.run(main)
