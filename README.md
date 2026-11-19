# HG-CrackSeg

HG-CrackSeg is a Hessian-guided network for pixel-wise fatigue crack segmentation
on steel structures. It combines a convolutional semantic encoder, large-kernel
residual blocks and ASPP with a differentiable multi-scale Hessian branch.
Geometric features guide the two highest-resolution decoder stages. Attention
gates filter the skip connections, and one two-channel head predicts background
and crack logits.

The model uses RGB images and is trained from scratch. It has no pretrained
backbone. The default C32 configuration contains
4,874,529 parameters. The repository includes training, testing, structural
ablations and the 18-condition Steelcrack-C robustness benchmark.

## Installation

Use Python 3.10. From the repository directory, create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
```

On Linux, activate with `source .venv/bin/activate`. The installation above
matches the timing environment used in the experiments. Training and evaluation
also support the tested PyTorch 2.5.1 + CUDA 11.8 configuration on Windows with
an RTX 2080 Ti:

```shell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

Install one matching pair, following the [PyTorch installation table](https://pytorch.org/get-started/previous-versions/).
FPS values are comparable only under the same hardware, software and measurement
settings. W&B is optional; training metrics are saved locally by default.

## Data

Download Steelcrack from the authors' [Civil-dataset repository](https://github.com/hzlbbfrog/Civil-dataset).
The experiments use the original images, masks and official split: 3,300 training,
525 validation and 530 test images, all 512 × 512.

```text
Steelcrack/
  Train/images/       Train/masks/
  Validation/images/  Validation/masks/
  Test/images/        Test/masks/
```

Images and masks must have matching filenames. Masks use 0 for background and
255 for cracks; the loader binarizes values greater than 127. See
[data preparation](docs/DATASETS.md) for the source citation and file manifest.

```shell
python tools/check_data.py --root Steelcrack
```

All configured relative paths resolve from the repository directory. Data may
remain elsewhere: edit `data.root_dir` or pass `--data-root` during training/testing.

## Training

```shell
python train.py --config configs/hg_crackseg.yaml --check
python train.py --config configs/hg_crackseg.yaml
```

The default recipe uses 100 epochs, batch size 8, AdamW, learning rate 0.0003,
weight decay 0.0001 and cosine annealing. The Hessian module is frozen for the
first five epochs and uses 0.008 times the base learning rate. The objective is
cross entropy plus Dice loss on differentiably hardened crack probabilities.
RGB inputs use the fixed ImageNet channel means and standard deviations.

The model with the highest validation image-wise Dice is saved as
`checkpoints/hg_crackseg/best_model.pth`. The same directory contains
`history.csv`, `config.yaml`, `best_metrics.json` and `last_checkpoint.pth`.

```shell
python train.py --config configs/hg_crackseg.yaml --resume checkpoints/hg_crackseg/last_checkpoint.pth
```

See [training details](docs/TRAINING.md) and [ablation configurations](docs/ABLATIONS.md).

## Testing

```shell
python test.py --config configs/hg_crackseg.yaml --checkpoint checkpoints/hg_crackseg/best_model.pth
```

Add `--save-visualizations` to export lossless prediction masks and
original/ground-truth/prediction panels. Add `--no-benchmark` to skip speed timing.
Results are written to `outputs/test/`. The default test directory is
`data.root_dir/Test`; `--data-root` instead accepts a directory containing
`images/` and `masks/` directly.

Metrics are means of per-image foreground Dice, IoU, clDice, precision and recall.
Predictions use crack probability greater than 0.5. `summary.csv` stores fractions;
`summary_percent.csv` stores percentages. Efficiency fields are parameters (M),
MACs (G), FLOPs (G), FPS (images/s) and latency (ms/image).

Checkpoints are loaded strictly against the selected model configuration. To
evaluate a trained model, supply its checkpoint using `--checkpoint`.

## Robustness

```shell
python generate_corruptions.py --root Steelcrack --out Steelcrack-C --seed 2026
python evaluate_robustness.py --config configs/robustness.yaml --check --deep-check
python evaluate_robustness.py --config configs/robustness.yaml
python plot_robustness.py --input outputs/robustness/summary.csv
```

This evaluates the same checkpoint on the clean test set and six corruptions
at three severity levels, for 19 test conditions in total.
Use `--checkpoint` to select another checkpoint and `--clean-root` /
`--corruption-root` to override the data locations. Repeating an evaluation
reuses complete results only when weights, data, code, settings and exported
artifacts match. Details and metric definitions are in
[the robustness protocol](docs/ROBUSTNESS.md).

## Efficiency

```shell
python benchmark.py --config configs/hg_crackseg.yaml --checkpoint checkpoints/hg_crackseg/best_model.pth
```

The benchmark loads the selected weights and measures FP32 inference with batch
size 1 and 512 × 512 inputs: 50 warm-up iterations and three repeats of 200 timed
forwards. CUDA synchronization brackets each timed repeat. Loading images,
transfers, postprocessing and file output are excluded. MACs include the
functional Hessian convolutions; FLOPs are reported as twice the estimated MACs.
The estimate does not count every scalar operation.

## Tests

```shell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover model/configuration compatibility, checkpoint loading,
structural ablations, metrics and mask export, corruption regeneration, and
epoch-boundary training resume.

## License

The source code is currently unlicensed. See the original dataset and dependency
repositories for their respective license terms.
