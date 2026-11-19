from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from data.normalization import normalization_parameters

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CORRUPTIONS = (
    "gaussian_noise",
    "brightness_shift",
    "uneven_light",
    "motion_blur",
    "jpeg_compression",
    "marker_distractor",
)


def dataset_specs(project_root: Path, settings: dict) -> list[dict]:

    def resolve(path):
        path = Path(path)
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()

    specs = []
    if settings.get("include_clean", True):
        specs.append(
            dict(
                dataset_id="clean",
                dataset_kind="clean",
                corruption="clean",
                severity=0,
                root=resolve(settings.get("clean_root", "Steelcrack/Test")),
                expected_count=settings.get("clean_count", 530),
            )
        )
    if settings.get("include_corruptions", True):
        corrupted = resolve(settings.get("corruption_root", "Steelcrack-C"))
        for corruption in CORRUPTIONS:
            for severity in (1, 2, 3):
                specs.append(
                    dict(
                        dataset_id=f"{corruption}/{severity}",
                        dataset_kind="corruption",
                        corruption=corruption,
                        severity=severity,
                        root=corrupted / corruption / str(severity),
                        expected_count=settings.get("corruption_count", 530),
                    )
                )
    return specs


def index_pairs(root: Path, expected_count: int | None = None) -> list[tuple[Path, Path]]:
    (image_dir, mask_dir) = (root / "images", root / "masks")
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ in {root}")
    images = {
        p.name: p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        p.name: p
        for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images or images.keys() != masks.keys():
        raise ValueError(
            f"Unpaired images/masks in {root}: missing masks={sorted(images.keys() - masks.keys())[:5]}, missing images={sorted(masks.keys() - images.keys())[:5]}"
        )
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"{root}: expected {expected_count} pairs, found {len(images)}")
    return [(images[name], masks[name]) for name in sorted(images)]


def fingerprint_pairs(pairs, *, content: bool = True) -> str:
    digest = hashlib.sha256()
    for image_path, mask_path in pairs:
        for role, path in (("image", image_path), ("mask", mask_path)):
            digest.update(f"{role}/{path.name}\x00".encode("utf-8"))
            if content:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                stat = path.stat()
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


class EvaluationDataset:

    def __init__(
        self, root, pairs=None, normalization="imagenet", image_size=512, require_positive=True
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        (self.mean, self.std) = normalization_parameters(normalization)
        self.pairs = pairs if pairs is not None else index_pairs(self.root)
        self.normalization = normalization
        self.image_size = int(image_size)
        self.require_positive = bool(require_positive)
        self.source_ids = {}
        mapping = self.root / "source_mapping.csv"
        if mapping.is_file():
            with mapping.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    self.source_ids[row["filename"]] = (
                        row.get("original_photo_name_from_filename") or row["filename"]
                    )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (image_path, mask_path) = self.pairs[index]
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L")) > 127
        expected = (self.image_size, self.image_size)
        if image.shape[:2] != expected or mask.shape != expected:
            raise ValueError(
                f"Unexpected image/mask dimensions: {image_path.name}: {image.shape}, {mask.shape}"
            )
        if self.require_positive and (not mask.any()):
            raise ValueError(f"Unexpected empty GT in this positive-only benchmark: {mask_path}")
        if self.normalization != "zero_one":
            image = (image - np.array(self.mean, dtype=np.float32)) / np.array(
                self.std, dtype=np.float32
            )
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return (
            image,
            np.ascontiguousarray(mask),
            image_path.name,
            self.source_ids.get(image_path.name, image_path.stem),
        )
