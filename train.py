"""Train HG-CrackSeg on Steelcrack."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.checkpoint import load_checkpoint
from core.losses import get_loss_function
from core.metrics import calculate_image_wise_metrics, extract_segmentation_logits
from data.evaluation_dataset import fingerprint_pairs, index_pairs
from data.steelcrack_dataset import SteelcrackDataset
from data.transforms import get_train_transforms, get_val_test_transforms
from models import create_model

ROOT = Path(__file__).resolve().parent
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, config):
    training = config["training"]
    lr = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    if hasattr(model, "hessian"):
        hessian = list(model.hessian.parameters())
        ids = {id(p) for p in hessian}
        groups = [
            {
                "params": [p for p in model.parameters() if id(p) not in ids],
                "lr": lr,
                "group_name": "base",
            },
            {
                "params": hessian,
                "lr": lr * float(training.get("ops_lr_scale", 0.008)),
                "group_name": "hessian",
            },
        ]
    else:
        groups = [{"params": list(model.parameters()), "lr": lr, "group_name": "base"}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def apply_hessian_freeze_schedule(model, epoch, freeze_epochs):
    if hasattr(model, "hessian"):
        for parameter in model.hessian.parameters():
            parameter.requires_grad = epoch > freeze_epochs


def run_epoch(model, loader, criterion, device, optimizer=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    metrics = {key: [] for key in ("iou", "dice", "precision", "recall", "acc")}
    losses = {}
    batches = 0
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, desc="Train" if training else "Validation", leave=False):
            images = images.to(device, non_blocking=True)
            masks = (masks > 0).float().to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, details = criterion(outputs, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nonfinite loss; no checkpoint selected from a partial epoch"
                )
            if training:
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm, error_if_nonfinite=True
                    )
                optimizer.step()
            image_metrics = calculate_image_wise_metrics(
                extract_segmentation_logits(outputs).detach().float(), masks, threshold=0.5
            )
            for key in metrics:
                metrics[key].extend(image_metrics[key].detach().cpu().tolist())
            for key, value in details.items():
                losses[key] = losses.get(key, 0.0) + float(value)
            batches += 1
    if not batches:
        raise ValueError("Empty epoch")
    return {
        **{f"mi_{k}": float(np.mean(v)) for k, v in metrics.items()},
        **{k: v / batches for k, v in losses.items()},
    }


def rng_state(generator):
    np_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
        torch=torch.get_rng_state(),
        loader=generator.get_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state, generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_history(output, history):
    temporary = output / "history.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, output / "history.csv")


def recipe(config):
    training = dict(config["training"])
    for key in ("save_dir", "init_weight", "wandb_mode", "save_interval", "device"):
        training.pop(key, None)
    data = dict(config["data"])
    data.pop("root_dir", None)
    return dict(model=config["model"], loss=config["loss"], data=data, training=training)


def runtime_state(device):
    return dict(
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "albumentations",
                "opencv-python-headless",
            )
        },
        cpu_threads=torch.get_num_threads(),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hg_crackseg.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Stop after this epoch; preserve the configured scheduler horizon",
    )
    parser.add_argument(
        "--check", action="store_true", help="Check configuration and paired data without training"
    )
    parser.add_argument("--resume", help="Full last_checkpoint.pth from this trainer")
    args = parser.parse_args(argv)
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8-sig"))
    training, data = config["training"], config["data"]
    if args.data_root:
        data["root_dir"] = args.data_root
    if args.device:
        training["device"] = args.device
    if data.get("normalization") != "imagenet":
        raise ValueError("data.normalization must be imagenet")
    if (
        training.get("optimizer", "AdamW") != "AdamW"
        or training.get("scheduler", "CosineAnnealingLR") != "CosineAnnealingLR"
    ):
        raise ValueError("The training recipe requires AdamW and CosineAnnealingLR")
    device = torch.device(training.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; select --device cpu explicitly or install CUDA PyTorch"
        )
    root = resolve(data["root_dir"])
    pairs = {split: index_pairs(root / split) for split in ("Train", "Validation")}
    epochs = int(training["epochs"])
    stop = args.epochs if args.epochs is not None else epochs
    if not 1 <= stop <= epochs or int(data["batch_size"]) < 2:
        raise ValueError("Require 1 <= --epochs <= training.epochs and batch_size >= 2")
    if any(len(p) % int(data["batch_size"]) == 1 for split, p in pairs.items() if split == "Train"):
        raise ValueError(
            "The last training batch would contain one image, incompatible with ASPP BatchNorm"
        )
    workers = int(data.get("num_workers", 0))
    if args.resume and workers != 0:
        raise ValueError("Exact epoch-boundary resume requires num_workers=0")
    set_random_seed(int(training.get("seed", 42)))
    model = create_model(config["model"])
    criterion = get_loss_function(config["loss"]["name"], **config["loss"].get("kwargs", {}))
    if args.check:
        print(
            f"Ready: Train={len(pairs['Train'])}, Validation={len(pairs['Validation'])}, "
            f"parameters={sum(p.numel() for p in model.parameters()):,}, device={device}"
        )
        return 0
    output = (
        resolve(args.output_dir)
        if args.output_dir
        else resolve(training["save_dir"]) / config["run_name"]
    )
    if not args.resume and any(
        (output / name).exists()
        for name in ("last_checkpoint.pth", "best_model.pth", "history.csv")
    ):
        raise FileExistsError(f"Existing run: {output}. Use --resume or a new --output-dir")
    fingerprints = {split: fingerprint_pairs(value) for split, value in pairs.items()}
    generator = torch.Generator().manual_seed(int(training.get("seed", 42)))
    common = dict(
        batch_size=int(data["batch_size"]),
        num_workers=workers,
        pin_memory=bool(data.get("pin_memory", False)),
        persistent_workers=bool(data.get("persistent_workers", False)) and workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        SteelcrackDataset(root, "Train", get_train_transforms("imagenet")),
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        SteelcrackDataset(root, "Validation", get_val_test_transforms("imagenet")),
        shuffle=False,
        **common,
    )
    model.to(device)
    criterion.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training.get("eta_min", 1e-6))
    )
    start, best_dice, best_epoch, best_state, history = 1, -float("inf"), 0, None, []
    if args.resume:
        checkpoint = torch.load(resolve(args.resume), map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "hg_crackseg_training_v1":
            raise ValueError("--resume requires last_checkpoint.pth, not inference weights")
        if checkpoint["recipe"] != recipe(config) or checkpoint["data_hashes"] != fingerprints:
            raise ValueError("Resume recipe/data differ from the saved run")
        if checkpoint["device_type"] != device.type or checkpoint["torch_version"] != str(
            torch.__version__
        ):
            raise ValueError("Resume requires the saved device type and PyTorch version")
        if checkpoint["runtime"] != runtime_state(device):
            raise ValueError("Resume requires matching libraries and numerical runtime settings")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        start = checkpoint["epoch"] + 1
        best_dice, best_epoch, best_state = (
            checkpoint["best_dice"],
            checkpoint["best_epoch"],
            checkpoint["best_model"],
        )
    elif training.get("init_weight"):
        load_checkpoint(model, resolve(training["init_weight"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger = None
    if training.get("wandb_mode", "disabled") != "disabled":
        import wandb

        logger = wandb.init(
            project=config.get("project_name", "HG-CrackSeg"),
            name=config["run_name"],
            config=config,
            mode=training["wandb_mode"],
        )
    if args.resume:
        restore_rng(checkpoint["rng"], generator)
        save_torch(output / "best_model.pth", best_state)
        write_history(output, history)
    print(f"Training on {device}; output: {output}", flush=True)
    try:
        for epoch in range(start, stop + 1):
            began = time.perf_counter()
            apply_hessian_freeze_schedule(
                model, epoch, int(training.get("operator_freeze_epochs", 5))
            )
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                float(training.get("gradient_clip_norm", 1.0)),
            )
            val_metrics = run_epoch(model, val_loader, criterion, device)
            row = dict(
                epoch=epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                **{f"lr/{g['group_name']}": g["lr"] for g in optimizer.param_groups},
                epoch_seconds=time.perf_counter() - began,
            )
            if val_metrics["mi_dice"] > best_dice:
                best_dice, best_epoch = val_metrics["mi_dice"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_torch(output / "best_model.pth", best_state)
            interval = int(training.get("save_interval", 5))
            if interval > 0 and epoch % interval == 0:
                save_torch(output / f"epoch_{epoch:03d}.pth", model.state_dict())
            scheduler.step()
            history.append(row)
            if logger:
                logger.log(row, step=epoch)
            state = dict(
                format="hg_crackseg_training_v1",
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                best_dice=best_dice,
                best_epoch=best_epoch,
                best_model=best_state,
                history=history,
                rng=rng_state(generator),
                recipe=recipe(config),
                data_hashes=fingerprints,
                device_type=device.type,
                torch_version=str(torch.__version__),
                runtime=runtime_state(device),
            )
            save_torch(output / "last_checkpoint.pth", state)
            write_history(output, history)
            print(
                f"Epoch {epoch}/{epochs}: loss={train_metrics['total_loss']:.4f}, "
                f"val Dice={val_metrics['mi_dice']:.4f}, val IoU={val_metrics['mi_iou']:.4f}; "
                f"best epoch={best_epoch}",
                flush=True,
            )
        (output / "best_metrics.json").write_text(
            json.dumps(
                dict(
                    epoch=best_epoch,
                    validation=(
                        {
                            k.removeprefix("val/"): v
                            for k, v in history[best_epoch - 1].items()
                            if k.startswith("val/")
                        }
                        if history
                        else {}
                    ),
                    checkpoint="best_model.pth",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        if logger:
            logger.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
