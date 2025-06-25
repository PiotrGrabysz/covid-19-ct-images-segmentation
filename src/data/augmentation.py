from dataclasses import dataclass

import albumentations
import cv2
from albumentations.pytorch import ToTensorV2


@dataclass
class AugmenentationConfig:
    horizontal_flip: bool = True
    elastic_transform: bool = True
    affine: bool = True
    brightness: bool = True
    noise: bool = True
    random_sized_crop: bool = True


def build_train_transforms(
    source_size: int,
    target_size: int,
    elastic_transform_strength: float=100,
    config: AugmenentationConfig = AugmenentationConfig(),
):
    transforms = []
    if config.horizontal_flip:
        transforms.append(albumentations.HorizontalFlip(p=0.5))

    if config.elastic_transform:
        transforms.append(
            albumentations.ElasticTransform(
                alpha=elastic_transform_strength,  # Strength of the distortion
                sigma=10,  # Smoothing
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            )
        )

    if config.affine:
        transforms.append(
            albumentations.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                translate_percent=(0.05, 0.05),
                shear=(-5, 5),
                p=0.7,
            )
        )

    if config.brightness:
        transforms.append(
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            )
        )

    if config.noise:
        transforms.append(albumentations.GaussNoise(std_range=(0.01, 0.05), p=0.5))

    if config.random_sized_crop:
        transforms.append(
            albumentations.RandomSizedCrop(
                min_max_height=(int(source_size * 0.75), source_size),
                size=(target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )
        )
    else:
        transforms.append(albumentations.Resize(
                target_size, target_size, interpolation=cv2.INTER_NEAREST
            ))

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
