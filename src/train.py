import os
from pathlib import Path

import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sagemaker.experiments.run import Run, load_run
from typing_extensions import Annotated

from src.data.data_loaders import build_data_loaders
from src.loss import build_loss
from src.model import UNet

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
    batch_size: Annotated[
        int, typer.Option(help="input batch size for training and testing")
    ] = 24,
    epoch: Annotated[int, typer.Option(help="Maximum number of epochs to train")] = 5,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.001,
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
    tensorboard_dir: Annotated[
        Path, typer.Option(help="directory where tensorboard saves logs)")
    ] = "/opt/ml/output/tensorboard",
    sagemaker_run_name: Annotated[
        str | None,
        typer.Option(
            help="name of the sagemaker experiment run. If it is not specified, one is auto generated."
        ),
    ] = None,
    checkpoint_dir: str | None = os.environ.get("SM_MODEL_DIR"),
    dry_run: Annotated[bool, typer.Option(help="quickly check a single pass")] = False,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    train_dataloader, test_dataloader = build_data_loaders(
        train=train,
        test=test,
        batch_size=batch_size,
        img_source_size=SOURCE_SIZE,
        img_target_size=TARGET_SIZE,
    )

    loss_fn = build_loss(alpha=alpha)

    model = UNet(
        metric_to_monitor=METRIC_TO_MONITOR,
        lr=lr,
        encoder_depth=encoder_depth,
        loss_fn=loss_fn,
    )

    tensorboard_logger = TensorBoardLogger(tensorboard_dir, name="covid_segmentation")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=METRIC_TO_MONITOR, mode="max", save_top_k=1, filename="best-model"
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
    print(f"Final {METRIC_TO_MONITOR}: {best_metric}")

    with Run(
        experiment_name="ct-images-segmentation", run_name=sagemaker_run_name
    ) as run:
        run.log_parameters(
            {
                "learning_rate": lr,
                "alpha": alpha,
                "encoder_depth": encoder_depth,
                "batch_size": batch_size,
                "epoch": epoch,
            }
        )
        # run.log_metric("final_accuracy", 0.87)
        # run.log_file("model_artifact.tar.gz", source="/opt/ml/model")


if __name__ == "__main__":
    typer.run(main)
