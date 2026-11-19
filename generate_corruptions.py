"""Generate the six Steelcrack-C corruptions at three severities."""

import argparse
import json
import math
import hashlib
import os
import platform
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(image_dir: Path):
    return sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0, 255).astype(np.uint8)


def adjust_brightness_gamma(img: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * alpha, 0.0, 1.0)
    x = np.power(x, gamma)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def add_uneven_light(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    (h, w) = img.shape[:2]
    (yy, xx) = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2.0) / max(w, 1)
    yy = (yy - h / 2.0) / max(h, 1)
    angle = rng.uniform(0, 2 * math.pi)
    grad = math.cos(angle) * xx + math.sin(angle) * yy
    grad = grad / (np.max(np.abs(grad)) + 1e-06)
    cx = rng.uniform(0.25 * w, 0.75 * w)
    cy = rng.uniform(0.25 * h, 0.75 * h)
    radius = rng.uniform(0.35, 0.65) * min(h, w)
    dist2 = ((np.arange(w)[None, :] - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2).astype(
        np.float32
    )
    spot = np.exp(-dist2 / (2.0 * radius * radius))
    spot = (spot - spot.min()) / (spot.max() - spot.min() + 1e-06)
    low = rng.normal(0, 1, size=(16, 16)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(low, (0, 0), sigmaX=35, sigmaY=35)
    low = low / (np.max(np.abs(low)) + 1e-06)
    illum = 1.0 + strength * 0.5 * grad + strength * 0.6 * (spot - 0.5) + strength * 0.25 * low
    low_bound = max(0.25, 1.0 - 0.85 * strength)
    high_bound = 1.0 + 0.95 * strength
    illum = np.clip(illum, low_bound, high_bound).astype(np.float32)
    x = img.astype(np.float32) * illum[:, :, None]
    return np.clip(x, 0, 255).astype(np.uint8)


def motion_blur_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    center = (k / 2.0 - 0.5, k / 2.0 - 0.5)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (k, k), flags=cv2.INTER_LINEAR)
    return kernel / (kernel.sum() + 1e-06)


def add_motion_blur(img: np.ndarray, kernel_size: int, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 180)
    kernel = motion_blur_kernel(kernel_size, angle)
    return cv2.filter2D(img, -1, kernel).astype(np.uint8)


def add_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    (ok, enc) = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def _mask_to_binary(mask_gray: np.ndarray) -> np.ndarray:
    if mask_gray is None:
        return None
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    return (mask_gray > 127).astype(np.uint8)


def _dilate_binary(mask_bin: np.ndarray, radius: int) -> np.ndarray:
    if mask_bin is None or radius <= 0:
        return mask_bin
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_bin.astype(np.uint8), kernel, iterations=1)


def _render_polyline_mask(h: int, w: int, pts: np.ndarray, width_px: int) -> np.ndarray:
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(
        line_mask,
        [pts.astype(np.int32)],
        isClosed=False,
        color=1,
        thickness=int(width_px),
        lineType=cv2.LINE_AA,
    )
    return (line_mask > 0).astype(np.uint8)


def _principal_crack_pose(mask_bin: np.ndarray, rng: np.random.Generator):
    (h, w) = mask_bin.shape[:2]
    (ys, xs) = np.where(mask_bin > 0)
    if len(xs) < 8:
        return (rng.uniform(0, 2 * math.pi), np.array([w / 2.0, h / 2.0], dtype=np.float32))
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = coords.mean(axis=0)
    coords0 = coords - center[None, :]
    cov = np.cov(coords0.T)
    (eigvals, eigvecs) = np.linalg.eigh(cov)
    main_vec = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.atan2(float(main_vec[1]), float(main_vec[0]))
    return (angle, center.astype(np.float32))


def _make_polyline(center, angle, total_len, n_points, bend_scale, h, w, rng):
    pts = []
    center = np.asarray(center, dtype=np.float32)
    dir_vec = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    norm_vec = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
    for i in range(int(n_points)):
        t = (i / max(n_points - 1, 1) - 0.5) * total_len
        bend = rng.normal(0, bend_scale * total_len)
        p = center + t * dir_vec + bend * norm_vec
        x = int(np.clip(p[0], 0, w - 1))
        y = int(np.clip(p[1], 0, h - 1))
        pts.append([x, y])
    return np.asarray(pts, dtype=np.int32)


def _sample_safe_center(
    h: int, w: int, protect_mask: np.ndarray, rng: np.random.Generator, margin: int = 20
):
    margin_x = min(max(margin, 1), max(w // 3, 1))
    margin_y = min(max(margin, 1), max(h // 3, 1))
    for _ in range(120):
        x = int(rng.integers(margin_x, max(margin_x + 1, w - margin_x)))
        y = int(rng.integers(margin_y, max(margin_y + 1, h - margin_y)))
        if protect_mask is None or protect_mask[y, x] == 0:
            return np.array([x, y], dtype=np.float32)
    if protect_mask is not None:
        (ys, xs) = np.where(protect_mask == 0)
        if len(xs) > 0:
            idx = int(rng.integers(0, len(xs)))
            return np.array([xs[idx], ys[idx]], dtype=np.float32)
    return np.array([w / 2.0, h / 2.0], dtype=np.float32)


def _sample_near_crack_parallel(mask_bin, protect_mask, severity, width_px, h, w, rng):
    (angle, crack_center) = _principal_crack_pose(mask_bin, rng)
    perp = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
    if severity == 1:
        offset_range = (18, 38)
        len_range = (0.28, 0.5)
        bend_scale = 0.018
    elif severity == 2:
        offset_range = (14, 48)
        len_range = (0.34, 0.62)
        bend_scale = 0.026
    else:
        offset_range = (10, 60)
        len_range = (0.38, 0.72)
        bend_scale = 0.035
    for _ in range(90):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        offset = rng.uniform(*offset_range) * sign
        jitter = rng.normal(0, 0.06 * max(h, w), size=2).astype(np.float32)
        center = crack_center + offset * perp + jitter
        center[0] = np.clip(center[0], 0, w - 1)
        center[1] = np.clip(center[1], 0, h - 1)
        angle2 = angle + rng.normal(0, math.radians(8))
        total_len = rng.uniform(*len_range) * max(h, w)
        pts = _make_polyline(
            center, angle2, total_len, int(rng.integers(2, 4)), bend_scale, h, w, rng
        )
        if _is_polyline_acceptable(pts, width_px, mask_bin, protect_mask, allow_overlap=False):
            return pts
    return None


def _sample_off_crack_line(mask_bin, protect_mask, severity, width_px, h, w, rng):
    if severity == 1:
        len_range = (0.18, 0.42)
        bend_scale = 0.02
        npts_range = (2, 3)
    elif severity == 2:
        len_range = (0.25, 0.55)
        bend_scale = 0.04
        npts_range = (2, 4)
    else:
        len_range = (0.3, 0.7)
        bend_scale = 0.06
        npts_range = (2, 5)
    for _ in range(140):
        center = _sample_safe_center(h, w, protect_mask, rng, margin=20)
        if rng.random() < 0.75:
            base_angles = [0, math.pi / 2, math.pi / 4, -math.pi / 4]
            angle = base_angles[int(rng.integers(0, len(base_angles)))] + rng.normal(
                0, math.radians(7)
            )
        else:
            angle = rng.uniform(0, math.pi)
        total_len = rng.uniform(*len_range) * max(h, w)
        n_points = int(rng.integers(npts_range[0], npts_range[1] + 1))
        pts = _make_polyline(center, angle, total_len, n_points, bend_scale, h, w, rng)
        if _is_polyline_acceptable(pts, width_px, mask_bin, protect_mask, allow_overlap=False):
            return pts
    return None


def _sample_rare_crossing_line(mask_bin, severity, width_px, h, w, rng):
    (angle, center) = _principal_crack_pose(mask_bin, rng)
    cross_angle = angle + rng.choice([-1.0, 1.0]) * rng.uniform(math.radians(35), math.radians(85))
    total_len = rng.uniform(0.18, 0.42) * max(h, w)
    pts = _make_polyline(center, cross_angle, total_len, int(rng.integers(2, 4)), 0.025, h, w, rng)
    return pts


def _is_polyline_acceptable(pts, width_px, mask_bin, protect_mask, allow_overlap=False) -> bool:
    (h, w) = mask_bin.shape[:2]
    line_mask = _render_polyline_mask(h, w, pts, width_px)
    if allow_overlap:
        return True
    if protect_mask is not None and np.any((line_mask > 0) & (protect_mask > 0)):
        return False
    return True


def _draw_polyline_blend(out, pts, width_px, alpha, color, blur_prob, rng):
    overlay = out.copy()
    cv2.polylines(
        overlay,
        [pts.astype(np.int32)],
        isClosed=False,
        color=color,
        thickness=int(width_px),
        lineType=cv2.LINE_AA,
    )
    if rng.random() < blur_prob:
        overlay = cv2.GaussianBlur(overlay, (3, 3), sigmaX=0.55)
    return (1.0 - alpha) * out + alpha * overlay


def add_marker_distractor(
    img: np.ndarray, severity: int, rng: np.random.Generator, mask_gray: np.ndarray = None
) -> np.ndarray:
    (h, w) = img.shape[:2]
    out = img.copy().astype(np.float32)
    mask_bin = _mask_to_binary(mask_gray)
    if mask_bin is None:
        mask_bin = np.zeros((h, w), dtype=np.uint8)
    elif mask_bin.shape[:2] != (h, w):
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_bin > 0).astype(np.uint8)
    if severity == 1:
        n_lines_range = (1, 2)
        width_range = (3, 6)
        alpha_range = (0.28, 0.45)
        clearance_radius = 12
        near_crack_prob = 0.55
        blur_prob = 0.45
        rare_cross_prob = 0.0
    elif severity == 2:
        n_lines_range = (2, 3)
        width_range = (4, 9)
        alpha_range = (0.35, 0.6)
        clearance_radius = 10
        near_crack_prob = 0.5
        blur_prob = 0.55
        rare_cross_prob = 0.0
    else:
        n_lines_range = (3, 5)
        width_range = (5, 14)
        alpha_range = (0.45, 0.75)
        clearance_radius = 7
        near_crack_prob = 0.42
        blur_prob = 0.65
        rare_cross_prob = 0.12
    protect_mask = _dilate_binary(mask_bin, clearance_radius)
    n_lines = int(rng.integers(n_lines_range[0], n_lines_range[1] + 1))
    color_choices = [(8, 8, 8), (25, 25, 25), (35, 25, 18), (18, 24, 50), (55, 22, 18)]
    allow_one_cross = severity == 3 and rng.random() < rare_cross_prob and np.any(mask_bin > 0)
    cross_has_been_used = False
    drawn = 0
    for line_idx in range(n_lines):
        width_px = int(rng.integers(width_range[0], width_range[1] + 1))
        alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))
        color = color_choices[int(rng.integers(0, len(color_choices)))]
        pts = None
        pts_is_cross = False
        if allow_one_cross and (not cross_has_been_used) and (rng.random() < 0.35):
            pts = _sample_rare_crossing_line(mask_bin, severity, width_px, h, w, rng)
            pts_is_cross = True
            cross_has_been_used = True
        else:
            if np.any(mask_bin > 0) and rng.random() < near_crack_prob:
                pts = _sample_near_crack_parallel(
                    mask_bin, protect_mask, severity, width_px, h, w, rng
                )
            if pts is None:
                pts = _sample_off_crack_line(mask_bin, protect_mask, severity, width_px, h, w, rng)
        if pts is None:
            continue
        if severity in (1, 2) and (
            not _is_polyline_acceptable(pts, width_px, mask_bin, protect_mask, allow_overlap=False)
        ):
            continue
        if severity == 3 and (not pts_is_cross):
            if not _is_polyline_acceptable(
                pts, width_px, mask_bin, protect_mask, allow_overlap=False
            ):
                continue
        out = _draw_polyline_blend(out, pts, width_px, alpha, color, blur_prob, rng)
        drawn += 1
    return np.clip(out, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "gaussian_noise": {1: {"sigma": 5}, 2: {"sigma": 10}, 3: {"sigma": 20}},
    "brightness_shift": {
        1: {"alpha": 0.8, "gamma": 1.1},
        2: {"alpha": 0.65, "gamma": 1.3},
        3: {"alpha": 0.5, "gamma": 1.6},
    },
    "uneven_light": {1: {"strength": 0.35}, 2: {"strength": 0.6}, 3: {"strength": 0.85}},
    "motion_blur": {1: {"kernel_size": 5}, 2: {"kernel_size": 9}, 3: {"kernel_size": 13}},
    "jpeg_compression": {1: {"quality": 70}, 2: {"quality": 50}, 3: {"quality": 30}},
    "marker_distractor": {
        1: {
            "severity": 1,
            "mask_aware": True,
            "overlap_policy": "no_overlap_with_dilated_crack_mask",
        },
        2: {
            "severity": 2,
            "mask_aware": True,
            "overlap_policy": "no_overlap_with_dilated_crack_mask",
        },
        3: {"severity": 3, "mask_aware": True, "overlap_policy": "rare_overlap_probability_0.12"},
    },
}


def apply_corruption(
    name: str,
    severity: int,
    img: np.ndarray,
    rng: np.random.Generator,
    mask_gray: np.ndarray = None,
) -> np.ndarray:
    params = CORRUPTIONS[name][severity]
    if name == "gaussian_noise":
        return add_gaussian_noise(img, sigma=params["sigma"], rng=rng)
    if name == "brightness_shift":
        return adjust_brightness_gamma(img, alpha=params["alpha"], gamma=params["gamma"])
    if name == "uneven_light":
        return add_uneven_light(img, strength=params["strength"], rng=rng)
    if name == "motion_blur":
        return add_motion_blur(img, kernel_size=params["kernel_size"], rng=rng)
    if name == "jpeg_compression":
        return add_jpeg_compression(img, quality=params["quality"])
    if name == "marker_distractor":
        return add_marker_distractor(img, severity=params["severity"], rng=rng, mask_gray=mask_gray)
    raise ValueError(f"Unknown corruption: {name}")


def read_image(path, flags):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Unable to decode: {path}")
    return image


def write_png(path, array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise OSError(f"Unable to encode: {path}")
    temporary = path.with_suffix(".png.tmp")
    encoded.tofile(temporary)
    os.replace(temporary, path)


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def find_mask(directory, name):
    direct = directory / name
    if direct.is_file():
        return direct
    matches = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.stem == Path(name).stem and p.suffix.lower() in IMG_EXTS
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one mask for {name}; found {len(matches)}")
    return matches[0]


def save_manifest(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def generate(args):
    project = Path(__file__).resolve().parent
    root, output = Path(args.root).expanduser(), Path(args.out).expanduser()
    root = (project / root).resolve() if not root.is_absolute() else root.resolve()
    output = (project / output).resolve() if not output.is_absolute() else output.resolve()
    if (
        root == output
        or root in output.parents
        or output in root.parents
        or output.parent == output
    ):
        raise ValueError("Input and output directories must be separate")
    images, masks = root / args.split / "images", root / args.split / "masks"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ under {root / args.split}")
    names = list_images(images)
    if not names or len({Path(n).stem for n in names}) != len(names):
        raise ValueError("Require nonempty images with unique filename stems")
    source = []
    for name in names:
        mask = find_mask(masks, name)
        source.append(
            dict(
                image=name,
                mask=mask.name,
                image_sha256=file_hash(images / name),
                mask_sha256=file_hash(mask),
            )
        )
    group_seeds = {
        f"{name}/{level}": int(args.seed + 1000 * level + 7919 * (index + 1))
        for index, name in enumerate(CORRUPTIONS)
        for level in (1, 2, 3)
    }
    protocol = dict(
        version="steelcrack_c_full18_v1",
        source_split=args.split,
        seed=args.seed,
        group_seeds=group_seeds,
        corruptions=CORRUPTIONS,
        sources=source,
        script_sha256=file_hash(Path(__file__)),
        python=platform.python_version(),
        numpy=np.__version__,
        opencv=cv2.__version__,
    )
    protocol = json.loads(json.dumps(protocol))
    manifest_path = output / "corruption_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise ValueError(
                "Nonempty output without a matching generation manifest; choose a new --out directory"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol") != protocol:
            raise ValueError(
                "Output protocol, inputs or software changed; choose a new --out directory"
            )
        unknown = (
            {p.name for p in output.iterdir()} - set(CORRUPTIONS) - {"corruption_manifest.json"}
        )
        if unknown:
            raise ValueError(f"Output contains unrelated entries: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        protocol=protocol,
        completed=False,
        outputs=[],
        masks="Original grayscale label pixels, encoded as PNG",
    )
    save_manifest(manifest_path, manifest)
    for name in CORRUPTIONS:
        for severity in (1, 2, 3):
            image_dir, mask_dir = (
                output / name / str(severity) / kind for kind in ("images", "masks")
            )
            for directory in (image_dir, mask_dir):
                directory.resolve().relative_to(output)
                directory.mkdir(parents=True, exist_ok=True)
                expected = {Path(n).with_suffix(".png").name for n in names}
                allowed = expected | {name + ".tmp" for name in expected}
                if {p.name for p in directory.iterdir()} - allowed:
                    raise ValueError(f"Unexpected files in {directory}")
            rng = np.random.default_rng(group_seeds[f"{name}/{severity}"])
            for item in tqdm(source, desc=f"{name}/{severity}"):
                image = cv2.cvtColor(
                    read_image(images / item["image"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                mask = read_image(masks / item["mask"], cv2.IMREAD_GRAYSCALE)
                if image.shape[:2] != mask.shape:
                    raise ValueError(f"Image/mask dimensions differ: {item['image']}")
                corrupted = apply_corruption(name, severity, image, rng, mask_gray=mask)
                basename = Path(item["image"]).with_suffix(".png").name
                destination_image, destination_mask = image_dir / basename, mask_dir / basename
                bgr = cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR)
                for destination, array, flags in (
                    (destination_image, bgr, cv2.IMREAD_COLOR),
                    (destination_mask, mask, cv2.IMREAD_GRAYSCALE),
                ):
                    destination.resolve().relative_to(output)
                    if destination.exists() and not args.overwrite:
                        if not np.array_equal(read_image(destination, flags), array):
                            raise ValueError(
                                f"Existing pixels differ: {destination}; use --overwrite to regenerate"
                            )
                    else:
                        write_png(destination, array)
                manifest["outputs"].append(
                    dict(
                        corruption=name,
                        severity=severity,
                        source=item["image"],
                        image=destination_image.relative_to(output).as_posix(),
                        mask=destination_mask.relative_to(output).as_posix(),
                        image_sha256=file_hash(destination_image),
                        mask_sha256=file_hash(destination_mask),
                    )
                )
    manifest["completed"] = True
    save_manifest(manifest_path, manifest)
    print(f"Generated 18 groups of {len(names)} pairs: {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Steelcrack")
    parser.add_argument("--out", default="Steelcrack-C")
    parser.add_argument("--split", default="Test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files in a matching managed output directory",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate(parse_args())
