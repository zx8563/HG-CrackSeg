"""Evaluate one validation-selected checkpoint on a paired test set."""

import argparse

from evaluate_robustness import read_yaml, resolve, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hg_crackseg.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", help="A directory containing images/ and masks/")
    parser.add_argument("--output-dir", default="outputs/test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    document = read_yaml(resolve(args.config))
    training = document["training"]
    checkpoint = args.checkpoint or str(
        resolve(training["save_dir"]) / document["run_name"] / "best_model.pth"
    )
    data_root = (
        resolve(args.data_root)
        if args.data_root
        else resolve(document["data"]["root_dir"]) / "Test"
    )
    entry = dict(
        id="hg_crackseg",
        name="HG-CrackSeg",
        config=args.config,
        checkpoint=checkpoint,
        normalization="imagenet",
    )
    spec = dict(
        dataset_id="clean",
        dataset_kind="clean",
        corruption="clean",
        severity=0,
        root=data_root,
        expected_count=args.expected_count,
    )
    config = dict(
        models=[entry],
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
        image_size=512,
        threshold=0.5,
        normalization="imagenet",
        require_positive=True,
        save_per_image=True,
        save_visualizations=args.save_visualizations,
        benchmark=dict(enabled=not args.no_benchmark, batch_size=1, warmup=50, runs=200, repeats=3),
    )
    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return run(config, [entry], [spec], output, args.device, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
