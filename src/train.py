import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_loaders import build_data_loaders
from src.model import UNet

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

    train_dataloader, test_dataloader = build_data_loaders(
        train=args.train,
        test=args.test,
        batch_size=args.batch_size,
        img_source_size=SOURCE_SIZE,
        img_target_size=TARGET_SIZE,
    )

    model = UNet(metric_to_monitor=METRIC_TO_MONITOR)

    trainer = Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        log_every_n_steps=5,
        fast_dev_run=True,
    )
    trainer.fit(model, train_dataloader, test_dataloader)


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
