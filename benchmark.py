"""Measure FP32 model-only complexity and inference speed."""

import argparse
import json

import torch

from core.benchmark import benchmark_model, profile_model
from core.checkpoint import load_checkpoint
from evaluate_robustness import (
    environment_info,
    hash_file,
    json_safe,
    read_yaml,
    resolve,
    write_csv,
    write_json,
)
from models import create_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hg_crackseg.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly")
    config = read_yaml(resolve(args.config))
    checkpoint = (
        resolve(args.checkpoint)
        if args.checkpoint
        else resolve(config["training"]["save_dir"]) / config["run_name"] / "best_model.pth"
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = create_model(config["model"], for_inference=True)
    loaded = load_checkpoint(model, checkpoint)
    model.to(device).float().eval()
    result = profile_model(model, device=device, architecture="hg_crackseg")
    result.update(
        benchmark_model(
            model, device=device, warmup=args.warmup, runs=args.runs, repeats=args.repeats
        )
    )
    result.update(
        model_name="HG-CrackSeg",
        checkpoint_sha256=hash_file(checkpoint),
        weights_loaded=True,
        checkpoint_loading=loaded,
        environment=environment_info(device),
    )
    output = resolve(args.output_dir)
    write_json(output / "benchmark.json", result)
    fields = [
        "model_name",
        "params_m",
        "macs_g",
        "flops_g",
        "fps",
        "latency_ms",
        "fps_std",
        "latency_ms_std",
        "profile_status",
        "benchmark_status",
    ]
    write_csv(output / "benchmark.csv", [result], fields)
    print(json.dumps(json_safe({k: result.get(k) for k in fields}), indent=2))
    return 0 if result["profile_status"] in {"partial", "estimated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
