from pathlib import Path
import csv
import hashlib
import os
import numpy as np
from PIL import Image, ImageDraw

ARTIFACT_FIELDS = (
    "filename",
    "predicted_mask",
    "comparison",
    "predicted_mask_sha256",
    "comparison_sha256",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def save_prediction_artifacts(destination, image_path, prediction, target):
    (destination, image_path) = (Path(destination), Path(image_path))
    (prediction, target) = (np.asarray(prediction), np.asarray(target))
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Prediction/GT must be equally sized two-dimensional masks")
    for mask in (prediction, target):
        if not np.logical_or(mask == 0, mask == 1).all():
            raise ValueError("Prediction/GT artifacts require binary 0/1 arrays")
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    (height, width) = prediction.shape
    if original.size != (width, height):
        raise ValueError(f"Image and prediction dimensions differ: {image_path}")
    predicted_image = Image.fromarray(prediction.astype(np.uint8) * 255)
    gt_image = Image.fromarray(target.astype(np.uint8) * 255).convert("RGB")
    output_name = image_path.name + ".png"
    mask_relative = Path("predicted_masks") / output_name
    panel_relative = Path("visualizations") / output_name
    header = 24
    panel = Image.new("RGB", (3 * width, height + header), "white")
    labels = ("Original", "Ground truth", "Prediction")
    draw = ImageDraw.Draw(panel)
    for index, (source, label) in enumerate(
        zip((original, gt_image, predicted_image.convert("RGB")), labels)
    ):
        panel.paste(source, (index * width, header))
        draw.text((index * width + 8, 6), label, fill="black")
    save_png(predicted_image, destination / mask_relative)
    save_png(panel, destination / panel_relative)
    return dict(
        filename=image_path.name,
        predicted_mask=mask_relative.as_posix(),
        comparison=panel_relative.as_posix(),
        predicted_mask_sha256=digest(destination / mask_relative),
        comparison_sha256=digest(destination / panel_relative),
    )


def artifacts_are_intact(destination, expected_hash, expected_count):
    destination = Path(destination).resolve()
    manifest = destination / "artifacts.csv"
    try:
        if not manifest.is_file() or digest(manifest) != expected_hash:
            return False
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        if (
            len(records) != expected_count
            or len({r["filename"] for r in records}) != expected_count
        ):
            return False
        for row in records:
            for key, folder in (
                ("predicted_mask", "predicted_masks"),
                ("comparison", "visualizations"),
            ):
                relative = Path(row[key])
                if (
                    relative.is_absolute()
                    or len(relative.parts) != 2
                    or relative.parts[0] != folder
                ):
                    return False
                path = (destination / relative).resolve()
                path.relative_to(destination)
                if not path.is_file() or digest(path) != row[key + "_sha256"]:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
