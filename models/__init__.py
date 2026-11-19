"""HG-CrackSeg model construction."""

from inspect import signature

from .hg_crackseg import HGCrackSeg

MODEL_REGISTRY = {"hg_crackseg": HGCrackSeg}
__all__ = ["HGCrackSeg", "create_model"]


def create_model(config_model, *, for_inference=False):
    if config_model.get("architecture", "hg_crackseg") != "hg_crackseg":
        raise ValueError("Expected architecture: hg_crackseg")
    allowed = set(signature(HGCrackSeg).parameters)
    unknown = set(config_model) - allowed - {"architecture", "img_size"}
    if unknown:
        raise ValueError(f"Unknown model options: {sorted(unknown)}")
    if config_model.get("in_channels", 3) != 3 or config_model.get("classes", 2) != 2:
        raise ValueError("HG-CrackSeg uses RGB input and two-class logits")
    return HGCrackSeg(**{k: v for k, v in config_model.items() if k in allowed})
