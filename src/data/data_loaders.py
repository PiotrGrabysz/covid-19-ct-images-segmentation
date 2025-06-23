from pathlib import Path

from torch.utils.data import DataLoader

from src.data.augmentation import (build_inference_transforms,
                                   build_train_transforms)
from src.data.dataset import NumpyDataset


def build_data_loaders(
    train: Path, test: Path, batch_size: int, img_source_size: int, img_target_size: int
) -> tuple[DataLoader, DataLoader]:
    train_transforms = build_train_transforms(img_source_size, img_target_size)
    train_dataset = NumpyDataset.from_path(train, transforms=train_transforms)

    inference_transforms = build_inference_transforms(img_target_size)
    test_dataset = NumpyDataset.from_path(test, transforms=inference_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
