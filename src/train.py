import argparse
import os
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    train_dataset = NumpyDataset.from_path(args.train)
    test_dataset = NumpyDataset.from_path(args.test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


class NumpyDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, transforms=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    @classmethod
    def from_path(cls, data_path: Path) -> Self:
        images = np.load(data_path / "images.npy").astype(np.float32)
        masks = np.load(data_path / "masks.npy").astype(np.int8)
        return NumpyDataset(images, masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        return image, mask


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
