"""Evaluate HG-CrackSeg on clean and corrupted images."""

from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parent
QUALITY = ("mi_dice", "mi_iou", "mi_cldice", "mi_precision", "mi_recall")
EFFICIENCY = ("params_m", "macs_g", "flops_g", "fps", "latency_ms")
FIELDS = [
    "model_id",
    "model_name",
    "dataset_id",
    "dataset_kind",
    "corruption",
    "severity",
    "num_images",
    "status",
    *QUALITY,
    *EFFICIENCY,
    "profile_status",
    "benchmark_status",
    "error",
]
PAPER_HEADERS = [
    "mi Dice",
    "mi IOU",
    "mi cldice",
    "mi_Precision",
    "mi_Recall",
    "Param. (M)",
    "MACs (G)",
    "FLOPs (G)",
    "FPS(images/s)",
    "Latency(ms/image)",
]


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_yaml(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows((json_safe(row) for row in rows))
    os.replace(temporary, path)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def code_signature():
    paths = [Path(__file__)]
    for folder in ("models", "core", "data"):
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    return signature({str(p.relative_to(ROOT)): hash_file(p) for p in paths})


def load_manifest(path):
    config = read_yaml(path)
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("models"), list)
        or (not config["models"])
    ):
        raise ValueError("Manifest requires a nonempty models list")
    ids = [m["id"] for m in config["models"]]
    if len(set(ids)) != len(ids) or any((not re.fullmatch("[a-z0-9_]+", mid) for mid in ids)):
        raise ValueError("Model IDs must be unique lowercase letters/digits/underscores")
    if int(config.get("image_size", 512)) != 512:
        raise ValueError("This benchmark uses native 512 x 512 inputs")
    if not 0 < float(config.get("threshold", 0.5)) < 1:
        raise ValueError("threshold must be between 0 and 1")
    if int(config.get("batch_size", 1)) < 1 or int(config.get("num_workers", 0)) < 0:
        raise ValueError("Invalid batch_size/num_workers")
    if int(config.get("benchmark", {}).get("batch_size", 1)) != 1:
        raise ValueError("Latency/FPS comparison requires benchmark.batch_size=1")
    return config


def select(items, requested, key):
    if not requested:
        return items
    unknown = set(requested) - {x[key] for x in items}
    if unknown:
        raise ValueError(f"Unknown {key}: {sorted(unknown)}")
    return [x for x in items if x[key] in requested]


def environment_info(device):
    info = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        device=str(device),
    )
    packages = {}
    for name in (
        "numpy",
        "Pillow",
        "PyYAML",
        "torch",
        "torchvision",
        "albumentations",
        "opencv-python-headless",
        "scikit-image",
        "thop",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info["packages"] = packages
    try:
        import torch

        info.update(
            cuda_available=torch.cuda.is_available(),
            torch_cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
        if str(device).startswith("cuda") and torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(torch.device(device))
            info.update(
                gpu_name=prop.name,
                gpu_total_memory=prop.total_memory,
                gpu_compute_capability=f"{prop.major}.{prop.minor}",
            )
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def validate_preprocessing(config, entry, model_document=None):
    from data.normalization import normalization_parameters

    normalization = entry.get("normalization", config.get("normalization", "imagenet"))
    normalization_parameters(normalization)
    saved_path = resolve(entry["checkpoint"]).parent / "config.yaml"
    if saved_path.is_file():
        training = read_yaml(saved_path)
        source = saved_path
        from inspect import signature
        from models import HGCrackSeg

        requested = model_document or read_yaml(resolve(entry["config"]))
        saved_model = training.get("model", {})
        for key, parameter in signature(HGCrackSeg).parameters.items():
            saved_value = saved_model.get(key, parameter.default)
            current_value = requested["model"].get(key, parameter.default)
            if key == "hessian_sigmas":
                saved_value, current_value = tuple(saved_value), tuple(current_value)
            if saved_value != current_value:
                raise ValueError(f"Model option {key} differs from the checkpoint's config.yaml")
    else:
        training = (
            model_document if model_document is not None else read_yaml(resolve(entry["config"]))
        )
        source = resolve(entry["config"])
    recorded = (training.get("data") or {}).get("normalization")
    if recorded is not None:
        normalization_parameters(recorded)
        if normalization != recorded:
            raise ValueError(
                f"{entry['id']}: inference normalization {normalization!r} does not match training normalization {recorded!r} recorded in {source}"
            )
    return normalization


def preflight(config, models, specs, device, deep=False, check_models=False, profile_only=False):
    from data.evaluation_dataset import EvaluationDataset, index_pairs

    report = dict(environment=environment_info(device), datasets=[], models=[], errors=[])
    for module in ("numpy", "PIL", "skimage.morphology", "sklearn.metrics", "torch", "thop"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            report["errors"].append(f"Dependency {module}: {exc}")
    if str(device).startswith("cuda") and (not report["environment"].get("cuda_available")):
        report["errors"].append(
            "CUDA is unavailable in this Python environment; there is no automatic CPU fallback"
        )
    for spec in specs:
        item = {k: str(v) if isinstance(v, Path) else v for (k, v) in spec.items()}
        try:
            pairs = index_pairs(spec["root"], spec["expected_count"])
            item.update(status="paired", num_images=len(pairs))
            if deep:
                dataset = EvaluationDataset(
                    spec["root"],
                    pairs,
                    image_size=config.get("image_size", 512),
                    require_positive=config.get("require_positive", True),
                )
                for i in range(len(dataset)):
                    dataset[i]
                item["status"] = "decoded_and_validated"
        except Exception as exc:
            item.update(status="error", error=str(exc))
            report["errors"].append(f"Dataset {spec['dataset_id']}: {exc}")
        report["datasets"].append(item)
    for entry in models:
        item = dict(
            model_id=entry["id"],
            config=str(resolve(entry["config"])),
            checkpoint=str(resolve(entry["checkpoint"])),
        )
        issues = []
        try:
            model_document = read_yaml(item["config"])
            if not profile_only:
                item["normalization"] = validate_preprocessing(config, entry, model_document)
            model_config = model_document["model"]
            item["architecture"] = model_config["architecture"]
            architecture = model_config["architecture"].lower()
            from models import MODEL_REGISTRY

            if architecture not in MODEL_REGISTRY:
                raise ValueError(f"Unsupported architecture: {architecture}")
            if not profile_only and (not Path(item["checkpoint"]).is_file()):
                issues.append("Checkpoint not found")
            if not profile_only and (not entry.get("preprocessing_confirmed", True)):
                issues.append(
                    "Training preprocessing remains unconfirmed (preprocessing_confirmed=false)"
                )
            if check_models:
                import torch
                from models import create_model
                from core.metrics import extract_segmentation_logits

                model = create_model(model_config, for_inference=True).to(device).eval()
                if not profile_only:
                    load_checkpoint(model, item["checkpoint"])
                with torch.inference_mode():
                    output = extract_segmentation_logits(
                        model(torch.zeros(1, 3, 512, 512, device=device))
                    )
                if tuple(output.shape[-2:]) != (512, 512) or not torch.isfinite(output).all():
                    raise ValueError(f"Invalid forward output {output.shape}")
                item["forward_shape"] = list(output.shape)
                item["forward_weights_loaded"] = not profile_only
                item["actual_params_m"] = sum((p.numel() for p in model.parameters())) / 1000000.0
                del output, model
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
        except Exception as exc:
            issues.append(str(exc))
        item.update(status="error" if issues else "ready", issues=issues)
        report["models"].append(item)
        report["errors"].extend((f"Model {entry['id']}: {issue}" for issue in issues))
    report["ready"] = not report["errors"]
    return report


from core.checkpoint import load_checkpoint


def base_row(entry, spec):
    return dict(
        model_id=entry["id"],
        model_name=entry.get("name", entry["id"]),
        **{k: spec[k] for k in ("dataset_id", "dataset_kind", "corruption", "severity")},
        num_images=0,
        status="pending",
    )


def write_summaries(output, rows, profiles):
    write_csv(output / "summary.csv", rows, FIELDS)
    paper_rows = []
    for row in rows:
        paper = {k: row.get(k) for k in FIELDS if k not in QUALITY + EFFICIENCY}
        for index, (key, label) in enumerate(zip(QUALITY + EFFICIENCY, PAPER_HEADERS)):
            value = row.get(key)
            paper[label] = value * 100 if value is not None and index < 5 else value
        paper_rows.append(paper)
    write_csv(
        output / "summary_percent.csv",
        paper_rows,
        [k for k in FIELDS if k not in QUALITY + EFFICIENCY] + PAPER_HEADERS,
    )
    write_csv(
        output / "model_profile.csv",
        profiles,
        [
            "model_id",
            "model_name",
            *EFFICIENCY,
            "profile_status",
            "profile_notes",
            "counting_convention",
            "fps_std",
            "latency_ms_std",
            "device",
            "benchmark_status",
        ],
    )
    if not any((row["dataset_kind"] == "corruption" for row in rows)):
        return
    aggregate = []
    for model_id in dict.fromkeys((row["model_id"] for row in rows)):
        selected = [r for r in rows if r["model_id"] == model_id]
        clean = next(
            (r for r in selected if r["dataset_kind"] == "clean" and r["status"] == "completed"),
            None,
        )
        corrupt = [r for r in selected if r["dataset_kind"] == "corruption"]
        groups = [("all_18", corrupt, 18)]
        from data.evaluation_dataset import CORRUPTIONS

        groups += [(c, [r for r in corrupt if r["corruption"] == c], 3) for c in CORRUPTIONS]
        for group, group_rows, expected in groups:
            complete = len(group_rows) == expected and all(
                (r["status"] == "completed" for r in group_rows)
            )
            result = dict(
                model_id=model_id,
                group=group,
                status="completed" if complete else "incomplete",
                completed_tests=sum((r["status"] == "completed" for r in group_rows)),
                expected_tests=expected,
            )
            if complete:
                for key in QUALITY:
                    mean = sum((r[key] for r in group_rows)) / expected
                    result[key] = mean
                    result[key + "_drop_pp"] = 100 * (clean[key] - mean) if clean else None
                    result[key + "_retention_pct"] = (
                        100 * mean / clean[key] if clean and clean[key] > 0 else None
                    )
            aggregate.append(result)
    write_csv(
        output / "robustness_summary.csv",
        aggregate,
        ["model_id", "group", "status", "completed_tests", "expected_tests"]
        + [name for key in QUALITY for name in (key, key + "_drop_pp", key + "_retention_pct")],
    )


def evaluate_task(model, device, config, entry, spec, pairs, destination):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from core.metrics import segmentation_probabilities, binary_image_metrics
    from data.evaluation_dataset import EvaluationDataset

    dataset = EvaluationDataset(
        spec["root"],
        pairs,
        normalization=entry.get("normalization", config.get("normalization", "imagenet")),
        image_size=config.get("image_size", 512),
        require_positive=config.get("require_positive", True),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
    )
    sums = np.zeros(5, dtype=np.float64)
    (count, records, artifacts, started) = (0, [], [], time.perf_counter())
    save_visualizations = bool(config.get("save_visualizations", False))
    if save_visualizations:
        from core.prediction_artifacts import save_prediction_artifacts, ARTIFACT_FIELDS
    image_paths = {image.name: image for (image, _) in pairs}
    with torch.inference_mode():
        for images, targets, filenames, source_ids in loader:
            probabilities = segmentation_probabilities(
                model(images.to(device, dtype=torch.float32, non_blocking=True))
            )
            if (
                tuple(probabilities.shape) != tuple(targets.shape)
                or not torch.isfinite(probabilities).all()
            ):
                raise ValueError(
                    f"Invalid probabilities/GT shape or non-finite output: {probabilities.shape}, {targets.shape}"
                )
            predictions = (probabilities > float(config.get("threshold", 0.5))).cpu().numpy()
            for pred, target, filename, source_id in zip(
                predictions, targets.numpy(), filenames, source_ids
            ):
                metrics = binary_image_metrics(pred, target, skeleton_backend="skimage")
                values = [metrics[key.removeprefix("mi_")] for key in QUALITY]
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite image metrics: {filename}")
                sums += values
                count += 1
                if save_visualizations:
                    artifacts.append(
                        save_prediction_artifacts(destination, image_paths[filename], pred, target)
                    )
                if config.get("save_per_image", True):
                    records.append(
                        dict(filename=filename, source_id=source_id, **dict(zip(QUALITY, values)))
                    )
            if count % 100 < len(images) or count == len(dataset):
                print(f"  {entry['id']} | {spec['dataset_id']}: {count}/{len(dataset)}", flush=True)
    if count != len(pairs):
        raise ValueError(f"Expected {len(pairs)} images, evaluated {count}")
    result = dict(zip(QUALITY, (sums / count).tolist()))
    result.update(
        num_images=count, status="completed", evaluation_seconds=time.perf_counter() - started
    )
    if config.get("save_per_image", True):
        path = destination / "per_image.csv"
        write_csv(path, records, ["filename", "source_id", *QUALITY])
        result.update(per_image_path=str(path), per_image_sha256=hash_file(path))
    if save_visualizations:
        artifact_path = destination / "artifacts.csv"
        write_csv(artifact_path, artifacts, ARTIFACT_FIELDS)
        result.update(
            save_visualizations=True,
            artifact_count=len(artifacts),
            artifacts_manifest_sha256=hash_file(artifact_path),
        )
    return result


def cached_task(path, task_signature):
    if not path.is_file():
        return None
    try:
        result = read_json(path)
        if result.get("signature") != task_signature or result.get("status") != "completed":
            return None
        if result.get("per_image_path"):
            table = Path(result["per_image_path"])
            if not table.is_file() or hash_file(table) != result["per_image_sha256"]:
                return None
        if result.get("save_visualizations"):
            from core.prediction_artifacts import artifacts_are_intact

            if not artifacts_are_intact(
                path.parent, result.get("artifacts_manifest_sha256"), result["num_images"]
            ):
                return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def profile_is_usable(profile, model_key, benchmark_enabled=True):
    return (
        profile.get("signature") == model_key
        and profile.get("profile_status") in {"partial", "estimated"}
        and (not benchmark_enabled or profile.get("benchmark_status") == "completed")
    )


def run(config, models, specs, output, device_name, force=False, profile_only=False):
    import numpy as np
    import torch
    from models import create_model
    from core.benchmark import profile_model, benchmark_model
    from data.evaluation_dataset import index_pairs, fingerprint_pairs

    device = torch.device(device_name)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA unavailable; activate the intended GPU Python environment")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = environment_info(device)
    code_hash = code_signature()
    protocol = dict(
        image_size=512,
        threshold=config.get("threshold", 0.5),
        batch_size=config.get("batch_size", 1),
        normalization=config.get("normalization", "imagenet"),
        require_positive=config.get("require_positive", True),
        save_per_image=config.get("save_per_image", True),
        save_visualizations=bool(config.get("save_visualizations", False)),
        artifact_format="lossless_png_original_gt_prediction_v1",
        seed=seed,
        skeleton_backend="skimage",
        aggregation="mean of per-image foreground metrics",
        precision="FP32",
        tf32=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        benchmark=config.get("benchmark", {}),
    )
    write_json(
        output / "run_manifest.json",
        dict(
            config=config,
            protocol=protocol,
            environment=environment,
            selected_models=[m["id"] for m in models],
            selected_datasets=[s["dataset_id"] for s in specs],
            code_sha256=code_hash,
        ),
    )
    (indexed, data_hashes, data_errors) = ({}, {}, {})
    if not profile_only:
        for spec in specs:
            try:
                print(f"Index/hash dataset: {spec['dataset_id']}", flush=True)
                pairs = index_pairs(spec["root"], spec["expected_count"])
                indexed[spec["dataset_id"]] = pairs
                data_hashes[spec["dataset_id"]] = fingerprint_pairs(pairs)
                mapping = spec["root"] / "source_mapping.csv"
                if mapping.is_file():
                    data_hashes[spec["dataset_id"]] = signature(
                        [data_hashes[spec["dataset_id"]], hash_file(mapping)]
                    )
            except Exception as exc:
                data_errors[spec["dataset_id"]] = str(exc)
    rows = [] if profile_only else [base_row(m, s) for m in models for s in specs]
    (profiles, failures) = ([], 0)
    if not profile_only:
        write_summaries(output, rows, profiles)
    for entry in models:
        model = None
        selected_rows = [r for r in rows if r["model_id"] == entry["id"]]
        try:
            if not profile_only and (not entry.get("preprocessing_confirmed", True)):
                raise ValueError("Confirm the training normalization before evaluation")
            model_document = read_yaml(resolve(entry["config"]))
            if not profile_only:
                validate_preprocessing(config, entry, model_document)
            model_config = model_document["model"]
            weight_hash = None if profile_only else hash_file(resolve(entry["checkpoint"]))
            model_key = signature(
                dict(
                    model_config=model_config,
                    checkpoint_sha256=weight_hash,
                    entry=entry,
                    code=code_hash,
                    protocol=protocol,
                    environment=environment,
                )
            )
            profile_path = (
                output
                / "profiles"
                / (entry["id"] + ("_untrained.json" if profile_only else ".json"))
            )
            try:
                cached_profile = (
                    read_json(profile_path) if profile_path.is_file() and (not force) else {}
                )
            except (OSError, ValueError):
                cached_profile = {}
            benchmark_enabled = config.get("benchmark", {}).get("enabled", True)
            usable_profile = profile_is_usable(cached_profile, model_key, benchmark_enabled)
            task_signatures = {
                s["dataset_id"]: signature([model_key, s, data_hashes.get(s["dataset_id"])])
                for s in specs
            }
            cached = {}
            if not force and (not profile_only):
                for spec in specs:
                    if spec["dataset_id"] not in data_errors:
                        cached[spec["dataset_id"]] = cached_task(
                            output / "tasks" / entry["id"] / spec["dataset_id"] / "metrics.json",
                            task_signatures[spec["dataset_id"]],
                        )
            need_model = (
                profile_only
                or force
                or (not usable_profile)
                or any(
                    (
                        not cached.get(s["dataset_id"])
                        for s in specs
                        if s["dataset_id"] not in data_errors
                    )
                )
            )
            if need_model:
                print(f"Build model: {entry['id']}", flush=True)
                model = create_model(model_config, for_inference=True)
                if not profile_only:
                    load_checkpoint(model, resolve(entry["checkpoint"]))
                model.to(device).float().eval()
            if force or profile_only or (not usable_profile):
                profile = profile_model(
                    model,
                    input_shape=(1, 3, 512, 512),
                    device=device,
                    architecture=model_config["architecture"],
                )
                bench_config = config.get("benchmark", {})
                profile["benchmark_status"] = "disabled"
                if bench_config.get("enabled", True):
                    try:
                        profile.update(
                            benchmark_model(
                                model,
                                device=device,
                                input_shape=(1, 3, 512, 512),
                                warmup=int(bench_config.get("warmup", 50)),
                                runs=int(bench_config.get("runs", 200)),
                                repeats=int(bench_config.get("repeats", 3)),
                            )
                        )
                        profile["benchmark_status"] = "completed"
                    except Exception as exc:
                        profile.update(
                            benchmark_status="error",
                            benchmark_error=str(exc),
                            fps=None,
                            latency_ms=None,
                        )
                profile.update(
                    model_id=entry["id"],
                    model_name=entry.get("name", entry["id"]),
                    device=str(device),
                    signature=model_key,
                    environment=environment,
                    protocol=protocol,
                    checkpoint_sha256=weight_hash,
                    weights_loaded=not profile_only,
                )
                write_json(profile_path, profile)
            else:
                profile = cached_profile
            profiles.append(profile)
            if not profile_is_usable(profile, model_key, benchmark_enabled):
                failures += 1
                print(
                    f"INCOMPLETE PROFILE {entry['id']}: {profile.get('profile_notes')}; benchmark={profile.get('benchmark_status')}",
                    file=sys.stderr,
                    flush=True,
                )
            if profile_only:
                write_csv(
                    output / "model_profile_untrained.csv",
                    profiles,
                    [
                        "model_id",
                        "model_name",
                        *EFFICIENCY,
                        "profile_status",
                        "profile_notes",
                        "benchmark_status",
                    ],
                )
                continue
            for row, spec in zip(selected_rows, specs):
                path = output / "tasks" / entry["id"] / spec["dataset_id"] / "metrics.json"
                try:
                    if spec["dataset_id"] in data_errors:
                        raise ValueError(data_errors[spec["dataset_id"]])
                    result = cached.get(spec["dataset_id"])
                    if result:
                        print(f"Resume cached: {entry['id']} | {spec['dataset_id']}", flush=True)
                    else:
                        result = dict(row)
                        result.update(
                            evaluate_task(
                                model,
                                device,
                                config,
                                entry,
                                spec,
                                indexed[spec["dataset_id"]],
                                path.parent,
                            )
                        )
                        result.update(
                            signature=task_signatures[spec["dataset_id"]],
                            data_sha256=data_hashes[spec["dataset_id"]],
                            checkpoint_sha256=weight_hash,
                        )
                    result.update(
                        {
                            k: profile.get(k)
                            for k in EFFICIENCY + ("profile_status", "benchmark_status")
                        }
                    )
                    write_json(path, result)
                    row.update(result)
                except Exception as exc:
                    failures += 1
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    write_json(path, row)
                    print(
                        f"ERROR {entry['id']} | {spec['dataset_id']}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                write_summaries(output, rows, profiles)
        except Exception as exc:
            failures += 1
            for row in selected_rows:
                if row["status"] != "completed":
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            write_json(
                output / "errors" / (entry["id"] + ".json"), dict(error=str(exc), model=entry)
            )
            print(f"ERROR {entry['id']}: {exc}", file=sys.stderr, flush=True)
            if not profile_only:
                write_summaries(output, rows, profiles)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(
        f"Finished: {sum((r['status'] == 'completed' for r in rows))}/{len(rows)} dataset tasks; {failures} errors; {output}",
        flush=True,
    )
    return 1 if failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--models", nargs="+", help="Model IDs in the manifest")
    parser.add_argument(
        "--datasets", nargs="+", help="clean, gaussian_noise/1, ..., marker_distractor/3"
    )
    parser.add_argument("--device", help="Default cuda:0; CPU only when explicitly selected")
    parser.add_argument("--output-dir", help="Override output_dir in the manifest")
    parser.add_argument(
        "--check", action="store_true", help="Preflight only; no checkpoint inference"
    )
    parser.add_argument(
        "--deep-check", action="store_true", help="Decode all images/masks during --check"
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="During --check, strictly load weights and run one forward",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Profile untrained architectures without checkpoint inference",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute selected tasks despite matching cache"
    )
    parser.add_argument("--checkpoint", help="Override the checkpoint for a single selected model")
    parser.add_argument(
        "--model-config", help="Override the architecture config for a single selected model"
    )
    parser.add_argument("--clean-root")
    parser.add_argument("--corruption-root")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_manifest(resolve(args.config))
    output = resolve(args.output_dir or config.get("output_dir", "robustness_results"))
    output.mkdir(parents=True, exist_ok=True)
    from data.evaluation_dataset import dataset_specs

    models = select(config["models"], args.models, "id")
    if not args.models:
        for entry in models:
            if not entry.get("enabled", True):
                print(
                    f"DEFERRED: {entry['id']}: {entry.get('disabled_reason', 'disabled in manifest')}"
                )
        models = [entry for entry in models if entry.get("enabled", True)]
    if args.checkpoint or args.model_config:
        if len(models) != 1:
            raise ValueError("Checkpoint/config overrides require exactly one selected model")
        if args.checkpoint:
            models[0]["checkpoint"] = args.checkpoint
        if args.model_config:
            models[0]["config"] = args.model_config
    for key in ("clean_root", "corruption_root"):
        if getattr(args, key):
            config.setdefault("datasets", {})[key] = getattr(args, key)
    if args.no_benchmark:
        config.setdefault("benchmark", {})["enabled"] = False
    if args.save_visualizations:
        config["save_visualizations"] = True
    specs = select(dataset_specs(ROOT, config.get("datasets", {})), args.datasets, "dataset_id")
    device = args.device or config.get("device", "cuda:0")
    if (args.deep_check or args.check_models) and (not args.check):
        raise ValueError("--deep-check/--check-models require --check")
    if args.check:
        result = preflight(
            config,
            models,
            [] if args.profile_only else specs,
            device,
            args.deep_check,
            args.check_models,
            args.profile_only,
        )
        write_json(output / "preflight.json", result)
        print(
            f"Datasets: {len(result['datasets'])}; models: {len(models)}; ready={result['ready']}"
        )
        for error in result["errors"]:
            print(f"  {error}")
        print(f"Report: {output / 'preflight.json'}")
        return 0 if result["ready"] else 1
    return run(config, models, specs, output, device, args.force, args.profile_only)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Finished tasks were saved; rerun the same command to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
