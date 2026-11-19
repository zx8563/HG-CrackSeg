from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.normalization import IMAGENET_MEAN, IMAGENET_STD, normalization_parameters


def get_train_transforms(normalization: str = "imagenet") -> A.Compose:
    (mean, std) = normalization_parameters(normalization)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )


def get_val_test_transforms(normalization: str = "imagenet") -> A.Compose:
    (mean, std) = normalization_parameters(normalization)
    return A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0), ToTensorV2()])


__all__ = ["get_train_transforms", "get_val_test_transforms"]
