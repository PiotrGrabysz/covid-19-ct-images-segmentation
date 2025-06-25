from dataclasses import dataclass

import albumentations
import cv2
from albumentations.pytorch import ToTensorV2


@dataclass
class AugmentationProbability:
    horizontal_flip: float = 0.5
    elastic_transform: float = 0.5
    affine: float = 0.7
    brightness: float = 0.5
    noise: float = 0.5
    random_sized_crop: int = 1


def build_train_transforms(
    source_size: int,
    target_size: int,
    elastic_transform_strength: float = 100,
    probabilities: AugmentationProbability = AugmentationProbability(),
):
    transforms = [
        albumentations.HorizontalFlip(p=probabilities.horizontal_flip),
        albumentations.ElasticTransform(
            alpha=elastic_transform_strength,  # Strength of the distortion
            sigma=10,  # Smoothing
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            p=probabilities.elastic_transform,
        ),
        albumentations.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            translate_percent=(0.05, 0.05),
            shear=(-5, 5),
            p=probabilities.affine,
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=probabilities.brightness
        ),
        albumentations.GaussNoise(std_range=(0.01, 0.05), p=probabilities.noise),
    ]

    if probabilities.random_sized_crop == 1:
        transforms.append(
            albumentations.RandomSizedCrop(
                min_max_height=(int(source_size * 0.75), source_size),
                size=(target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )
        )
    else:
        transforms.append(
            albumentations.Resize(
                target_size, target_size, interpolation=cv2.INTER_NEAREST
            )
        )

    transforms.append(ToTensorV2())

    return albumentations.Compose(transforms)


def build_inference_transforms(target_size: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(
                target_size, target_size, interpolation=cv2.INTER_NEAREST
            ),
            ToTensorV2(),
        ]
    )
