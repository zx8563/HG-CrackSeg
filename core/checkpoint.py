"""Strict loading of current and legacy state dictionaries."""

from collections.abc import Mapping
from pathlib import Path

import torch

LEGACY_PREFIXES = {
    "down3_mamba.": "down3_large_kernel.",
    "down4_mamba.": "down4_large_kernel.",
}


def normalize_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a state dictionary or a checkpoint mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), Mapping):
            checkpoint = checkpoint[key]
            break
    state, renamed = {}, {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError("State dictionary must contain named tensors")
        name = key.removeprefix("module.")
        for old, new in LEGACY_PREFIXES.items():
            if name.startswith(old):
                name = new + name[len(old) :]
                break
        if name in state:
            raise ValueError(f"Duplicate normalized checkpoint key: {name}")
        state[name] = value
        if name != key:
            renamed[key] = name
    return state, renamed


def load_checkpoint(model, path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    state, renamed = normalize_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    return {"strict": True, "tensor_count": len(state), "renamed_keys": renamed}
