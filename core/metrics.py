from __future__ import annotations
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "calculate_image_wise_metrics",
    "calculate_image_wise_cldice",
    "calculate_image_wise_auc",
    "get_model_complexity",
    "extract_segmentation_logits",
    "segmentation_probabilities",
    "binary_image_metrics",
]


def extract_segmentation_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "body_logits" in outputs:
            outputs = outputs["body_logits"]
        elif "out" in outputs:
            outputs = outputs["out"]
        else:
            raise KeyError(f"No body_logits/out segmentation head: {list(outputs)}")
    if not torch.is_tensor(outputs):
        raise TypeError(f"Unsupported segmentation output: {type(outputs).__name__}")
    if outputs.ndim != 4 or outputs.size(1) not in (1, 2):
        raise ValueError(f"Expected [B,1|2,H,W] logits, got {tuple(outputs.shape)}")
    return outputs


def segmentation_probabilities(logits: torch.Tensor) -> torch.Tensor:
    logits = extract_segmentation_logits(logits)
    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    return F.softmax(logits, dim=1)[:, 1]


def _get_crack_probs_and_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = segmentation_probabilities(logits)
    if targets.ndim == 4:
        if targets.size(1) != 1:
            raise ValueError(
                f"4-D targets must have one channel, got shape {tuple(targets.shape)}."
            )
        targets = targets[:, 0]
    elif targets.ndim != 3:
        raise ValueError(
            f"Expected targets with shape [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}."
        )
    if tuple(probs.shape) != tuple(targets.shape):
        raise ValueError(f"Prediction/target shape mismatch: {probs.shape}, {targets.shape}")
    return (probs, targets.float())


def calculate_image_wise_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-06
) -> dict[str, torch.Tensor]:
    (probs, targets) = _get_crack_probs_and_targets(logits, targets)
    preds = (probs > threshold).float()
    preds_flat = preds.contiguous().view(preds.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    pred_sum = preds_flat.sum(dim=1)
    gt_sum = targets_flat.sum(dim=1)
    union = pred_sum + gt_sum - intersection
    iou = intersection / (union + eps)
    dice = 2.0 * intersection / (pred_sum + gt_sum + eps)
    recall = intersection / (gt_sum + eps)
    precision = intersection / (pred_sum + eps)
    correct = (preds_flat == targets_flat).float().sum(dim=1)
    accuracy = correct / targets_flat.size(1)
    return {"iou": iou, "dice": dice, "recall": recall, "precision": precision, "acc": accuracy}


def _skeletonize_binary(mask: np.ndarray, backend: str = "auto") -> np.ndarray:
    if backend not in {"auto", "skimage"}:
        raise ValueError(f"Unsupported skeleton backend: {backend!r}")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Skeletonization expects a 2-D mask, got {binary.shape}.")
    if binary.size == 0:
        raise ValueError("A skeletonization mask must not be empty.")
    if backend == "skimage":
        from skimage.morphology import skeletonize

        return skeletonize(binary.astype(bool))
    if binary.max() == 0:
        return binary.astype(bool)
    try:
        from skimage.morphology import skeletonize

        return skeletonize(binary.astype(bool))
    except Exception:
        pass
    try:
        import cv2

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            thinned = cv2.ximgproc.thinning(
                binary * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            return thinned > 0
    except Exception:
        pass
    try:
        import cv2

        work = binary * 255
        skeleton = np.zeros_like(work)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while cv2.countNonZero(work) > 0:
            eroded = cv2.erode(work, element)
            opened = cv2.dilate(eroded, element)
            residual = cv2.subtract(work, opened)
            skeleton = cv2.bitwise_or(skeleton, residual)
            work = eroded
        return skeleton > 0
    except Exception as exc:
        raise RuntimeError(
            "clDice skeletonization requires opencv-contrib-python, scikit-image, or OpenCV."
        ) from exc


def _binary_cldice(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-06,
    *,
    skeleton_backend: str = "auto",
) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    pred_any = bool(pred.any())
    target_any = bool(target.any())
    if not pred_any and (not target_any):
        return 1.0
    if pred_any != target_any:
        return 0.0
    pred_skel = _skeletonize_binary(pred, backend=skeleton_backend)
    target_skel = _skeletonize_binary(target, backend=skeleton_backend)
    pred_skel_sum = float(pred_skel.sum())
    target_skel_sum = float(target_skel.sum())
    if pred_skel_sum <= 0.0 and target_skel_sum <= 0.0:
        return 1.0
    if pred_skel_sum <= 0.0 or target_skel_sum <= 0.0:
        return 0.0
    topology_precision = float(np.logical_and(pred_skel, target).sum()) / (pred_skel_sum + eps)
    topology_sensitivity = float(np.logical_and(target_skel, pred).sum()) / (target_skel_sum + eps)
    denom = topology_precision + topology_sensitivity
    if denom <= eps:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / (denom + eps))


def calculate_image_wise_cldice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-06,
    *,
    skeleton_backend: str = "auto",
) -> list[float]:
    if skeleton_backend == "skimage":
        from skimage.morphology import skeletonize
    elif skeleton_backend != "auto":
        raise ValueError(f"Unsupported skeleton backend: {skeleton_backend!r}")
    (probs, targets) = _get_crack_probs_and_targets(logits, targets)
    preds = (probs > threshold).detach().cpu().numpy().astype(bool)
    targets_np = (targets > 0.5).detach().cpu().numpy().astype(bool)
    return [
        _binary_cldice(pred, target, eps=eps, skeleton_backend=skeleton_backend)
        for (pred, target) in zip(preds, targets_np)
    ]


def binary_image_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-06,
    *,
    skeleton_backend: str = "skimage",
) -> dict[str, float]:
    pred = np.asarray(pred_mask)
    target = np.asarray(target_mask)
    if pred.ndim != 2 or target.shape != pred.shape or (not pred.size):
        raise ValueError(
            f"Expected equally shaped nonempty 2-D masks: {pred.shape}, {target.shape}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")
    for name, array in (("prediction", pred), ("target", target)):
        if not np.isfinite(array).all() or not np.logical_or(array == 0, array == 1).all():
            raise ValueError(f"{name} must contain only finite binary 0/1 values")
    if skeleton_backend == "skimage":
        from skimage.morphology import skeletonize
    elif skeleton_backend != "auto":
        raise ValueError(f"Unsupported skeleton backend: {skeleton_backend!r}")
    (pred, target) = (pred.astype(bool, copy=False), target.astype(bool, copy=False))
    intersection = float(np.count_nonzero(pred & target))
    (pred_sum, gt_sum) = (float(pred.sum()), float(target.sum()))
    return {
        "dice": 2.0 * intersection / (pred_sum + gt_sum + eps),
        "iou": intersection / (pred_sum + gt_sum - intersection + eps),
        "cldice": _binary_cldice(pred, target, eps, skeleton_backend=skeleton_backend),
        "precision": intersection / (pred_sum + eps),
        "recall": intersection / (gt_sum + eps),
        "acc": float(np.count_nonzero(pred == target)) / pred.size,
    }


def calculate_image_wise_auc(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[list[float], list[float]]:
    (probs, targets) = _get_crack_probs_and_targets(logits, targets)
    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    probs_flat = probs_np.reshape(probs_np.shape[0], -1)
    targets_flat = targets_np.reshape(targets_np.shape[0], -1)
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    for y_true, y_score in zip(targets_flat, probs_flat):
        if np.unique(y_true).size == 2:
            roc_aucs.append(float(roc_auc_score(y_true, y_score)))
            pr_aucs.append(float(average_precision_score(y_true, y_score)))
        else:
            roc_aucs.append(float("nan"))
            pr_aucs.append(float("nan"))
    return (roc_aucs, pr_aucs)


def get_model_complexity(
    model: torch.nn.Module, input_res: tuple[int, int, int] = (3, 512, 512)
) -> tuple[float, float, float]:
    from .benchmark import profile_model

    result = profile_model(model, input_shape=(1, *input_res))
    if result["profile_status"] != "estimated":
        warnings.warn("; ".join(result["profile_notes"]), stacklevel=2)
    return (result["params_m"], result["macs_g"], result["flops_g"])
