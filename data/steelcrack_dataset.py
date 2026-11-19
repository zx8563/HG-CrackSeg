from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SteelcrackDataset(Dataset):
    VALID_SPLITS = {"Train", "Validation", "Test"}
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "Train",
        transform: Optional[Callable] = None,
        return_name: bool = False,
    ) -> None:
        super().__init__()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}, got {split!r}.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_name = bool(return_name)
        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        self.image_names = sorted(
            (
                path.name
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
            )
        )
        if not self.image_names:
            raise RuntimeError(f"No supported images found in {self.image_dir}")
        missing_masks = [name for name in self.image_names if not (self.mask_dir / name).is_file()]
        if missing_masks:
            preview = missing_masks[:10]
            raise FileNotFoundError(
                f"Masks with matching filenames are missing for {len(missing_masks)} image(s); examples: {preview}"
            )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        image_path = self.image_dir / img_name
        mask_path = self.mask_dir / img_name
        image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.asarray(image).transpose(2, 0, 1)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.asarray(mask))
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(f"Mask must have shape [H,W] or [1,H,W], got {tuple(mask.shape)}")
        if self.return_name:
            return (image, mask, img_name)
        return (image, mask)


__all__ = ["SteelcrackDataset"]
