"""Validate paired Steelcrack data and optionally export a file manifest."""

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.evaluation_dataset import index_pairs


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Steelcrack")
    parser.add_argument("--manifest", help="CSV output with relative filenames and SHA-256")
    args = parser.parse_args(argv)
    root = Path(args.root)
    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    rows = []
    for split, count in (("Train", 3300), ("Validation", 525), ("Test", 530)):
        pairs = index_pairs(root / split, count)
        for image, mask in pairs:
            with Image.open(image) as opened:
                if opened.size != (512, 512) or opened.mode != "RGB":
                    raise ValueError(f"Expected 512x512 RGB: {image}")
            with Image.open(mask) as opened:
                labels = np.asarray(opened.convert("L"))
            if labels.shape != (512, 512) or not (labels > 127).any():
                raise ValueError(
                    f"Expected 512x512 positive crack mask with foreground >127: {mask}"
                )
            if args.manifest:
                rows.append(
                    dict(
                        split=split,
                        image=image.relative_to(root).as_posix(),
                        mask=mask.relative_to(root).as_posix(),
                        image_sha256=digest(image),
                        mask_sha256=digest(mask),
                    )
                )
        print(f"{split}: {count} pairs validated")
    if args.manifest:
        output = Path(args.manifest)
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
