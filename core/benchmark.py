from __future__ import annotations
import copy
import importlib
import importlib.metadata
import math
import statistics
import time
from typing import Callable
import torch
from torch import nn


def _input_shape(shape) -> tuple[int, int, int, int]:
    shape = tuple((int(value) for value in shape))
    if len(shape) != 4 or any((value <= 0 for value in shape)):
        raise ValueError(f"Expected positive BCHW input_shape, got {shape}")
    if shape[0] != 1:
        raise ValueError("Complexity and latency measurements require batch size 1.")
    return shape


def _device_for(model: nn.Module, requested=None) -> torch.device:
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    device = (
        torch.device(requested)
        if requested is not None
        else tensor.device if tensor is not None else torch.device("cpu")
    )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported benchmark device: {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select device='cpu' or install CUDA-enabled PyTorch.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _AtomicProfileModule(nn.Module):

    def __init__(self, wrapped: nn.Module, rule: Callable) -> None:
        super().__init__()
        object.__setattr__(self, "wrapped", wrapped)
        object.__setattr__(self, "rule", rule)

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "wrapped"), name)


def _count_atomic(module, inputs, output) -> None:
    module.total_ops += float(module.rule(module.wrapped, inputs, output))


def _hessian_macs(module, inputs, output) -> float:
    (batch, _, height, width) = inputs[0].shape
    return float(batch * height * width * 3 * module.num_scales * module.max_kernel_size**2)


def _wrap_special_modules(module, notes, rules_used):
    from models.hg_crackseg import DifferentiableHessian

    if isinstance(module, DifferentiableHessian):
        rules_used.add(
            "Hessian: three functional convolutions; scalar vesselness operations excluded"
        )
        return _AtomicProfileModule(module, _hessian_macs)
    for name, child in list(module.named_children()):
        module._modules[name] = _wrap_special_modules(child, notes, rules_used)
    return module


def profile_model(
    model: nn.Module, input_shape=(1, 3, 512, 512), device=None, architecture: str | None = None
) -> dict:
    shape = _input_shape(input_shape)
    result = {
        "params_m": sum((parameter.numel() for parameter in model.parameters())) / 1000000.0,
        "macs_g": math.nan,
        "flops_g": math.nan,
        "profile_status": "unavailable",
        "profile_notes": [],
        "unsupported_modules": [],
        "custom_rules": [],
        "input_shape": list(shape),
        "architecture": architecture or type(model).__name__,
        "counting_convention": "THOP MAC estimate; Hessian functional convolutions included; FLOPs = 2 * MACs; not all scalar operations",
    }
    try:
        thop_module = importlib.import_module("thop.profile")
        profile = thop_module.profile
        register_hooks = thop_module.register_hooks
    except (ImportError, AttributeError) as exc:
        result["profile_notes"].append(f"Compatible THOP is unavailable: {exc}")
        return result
    for distribution in ("ultralytics-thop", "thop"):
        try:
            result["profiler_version"] = (
                f"{distribution}=={importlib.metadata.version(distribution)}"
            )
            break
        except importlib.metadata.PackageNotFoundError:
            pass
    copied = None
    try:
        target_device = _device_for(model, device)
        copied = copy.deepcopy(model).to(device=target_device, dtype=torch.float32).eval()
        rules_used: set[str] = set()
        copied = _wrap_special_modules(copied, result["profile_notes"], rules_used)
        result["custom_rules"] = sorted(rules_used)
        ignored = (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Flatten, nn.ReLU, nn.ReLU6)
        unsupported = set()
        for child in copied.modules():
            if isinstance(child, (_AtomicProfileModule,)) or isinstance(child, ignored):
                continue
            if not list(child.children()) and (
                not any((cls in register_hooks for cls in type(child).__mro__))
            ):
                unsupported.add(f"{type(child).__module__}.{type(child).__name__}")
        result["unsupported_modules"] = sorted(unsupported)
        dummy = torch.zeros(shape, device=target_device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(device_type=target_device.type, enabled=False):
            (macs, _) = profile(
                copied,
                inputs=(dummy,),
                custom_ops={_AtomicProfileModule: _count_atomic},
                verbose=False,
            )
        if not math.isfinite(float(macs)) or float(macs) <= 0:
            raise RuntimeError(f"Profiler produced invalid operation count: {macs}")
        result["macs_g"] = float(macs) / 1000000000.0
        result["flops_g"] = 2.0 * result["macs_g"]
        result["profile_status"] = "partial"
        result["profile_notes"].append(
            "Module-based estimate: functional interpolation, elementwise operations and custom kernel details are not fully covered."
        )
        result["profile_notes"].extend(sorted(rules_used))
        if unsupported:
            result["profile_notes"].append(
                "Unsupported leaf modules: " + ", ".join(sorted(unsupported))
            )
    except Exception as exc:
        result["profile_status"] = "failed"
        result["macs_g"] = result["flops_g"] = math.nan
        result["profile_notes"].append(f"{type(exc).__name__}: {exc}")
    finally:
        del copied
    return result


def benchmark_model(
    model: nn.Module,
    device=None,
    input_shape=(1, 3, 512, 512),
    warmup: int = 50,
    runs: int = 200,
    repeats: int = 3,
) -> dict:
    shape = _input_shape(input_shape)
    if warmup < 0 or runs <= 0 or repeats <= 0:
        raise ValueError("warmup >= 0, runs > 0 and repeats > 0 are required.")
    target_device = _device_for(model, device)
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device != target_device:
            raise ValueError(f"Model tensor is on {tensor.device}, expected {target_device}.")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("FP32 benchmark requires FP32 model parameters and buffers.")
    modes = {module: module.training for module in model.modules()}
    measurements = []
    try:
        model.eval()
        generator = torch.Generator(device=target_device).manual_seed(0)
        dummy = torch.randn(shape, device=target_device, dtype=torch.float32, generator=generator)
        with torch.inference_mode(), torch.autocast(device_type=target_device.type, enabled=False):
            for _ in range(warmup):
                model(dummy)
            for _ in range(repeats):
                _sync(target_device)
                start = time.perf_counter()
                for _ in range(runs):
                    output = model(dummy)
                _sync(target_device)
                elapsed = time.perf_counter() - start
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError(f"Invalid benchmark duration: {elapsed}")
                from .metrics import extract_segmentation_logits

                logits = extract_segmentation_logits(output)
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("Benchmark model returned nonfinite logits.")
                measurements.append(elapsed)
    finally:
        for module, training in modes.items():
            module.training = training
    latencies = [elapsed * 1000.0 / runs for elapsed in measurements]
    speeds = [runs / elapsed for elapsed in measurements]
    latency = statistics.mean(latencies)
    result = {
        "fps": 1000.0 / latency,
        "latency_ms": latency,
        "fps_std": statistics.pstdev(speeds),
        "latency_ms_std": statistics.pstdev(latencies),
        "fps_per_repeat": speeds,
        "latency_ms_per_repeat": latencies,
        "elapsed_seconds_per_repeat": measurements,
        "benchmark_status": "completed",
        "benchmark_scope": "model-level reference; forward-only; reused across datasets",
        "timing_method": "perf_counter with CUDA synchronization before/after each repeat",
        "input_shape": list(shape),
        "precision": "float32",
        "warmup": int(warmup),
        "runs": int(runs),
        "repeats": int(repeats),
        "device": str(target_device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    if target_device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(target_device),
                "cudnn_version": torch.backends.cudnn.version(),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            }
        )
    return result
