from pathlib import Path

from torch.utils.data import DataLoader

from src.data.augmentation import (AugmentationProbability,
                                   build_inference_transforms,
                                   build_train_transforms)
from src.data.dataset import NumpyDataset


def build_data_loaders(
    train: Path,
    test: Path,
    train_batch_size: int,
    test_batch_size: int,
    img_source_size: int,
    img_target_size: int,
    elastic_transform_strength: float = 100,
    augmentation_probability: AugmentationProbability = AugmentationProbability(),
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_transforms = build_train_transforms(
        img_source_size,
        img_target_size,
        probabilities=augmentation_probability,
        elastic_transform_strength=elastic_transform_strength,
    )
    train_dataset = NumpyDataset.from_path(train, transforms=train_transforms)

    inference_transforms = build_inference_transforms(img_target_size)
    test_dataset = NumpyDataset.from_path(test, transforms=inference_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, test_dataloader
