import albumentations
import cv2
from albumentations.pytorch import ToTensorV2


def build_train_transforms(source_size: int, target_size: int):
    return albumentations.Compose(
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
                p=0.7,
            ),
            albumentations.RandomSizedCrop(
                min_max_height=(int(source_size * 0.75), source_size),
                size=(target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            ),
            ToTensorV2(),
        ]
    )


def build_inference_transforms(target_size: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(
                target_size, target_size, interpolation=cv2.INTER_NEAREST
            ),
            ToTensorV2(),
        ]
    )
