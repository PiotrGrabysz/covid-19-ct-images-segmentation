from pathlib import Path
from typing import Self

import albumentations
import numpy as np
import torch
from torch.utils.data import Dataset


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

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        try:
            mask = mask.permute(2, 0, 1).double()
        except AttributeError:
            pass

        return image, mask
