# Steelcrack-C protocol

The benchmark contains the clean 530-image test set and 18 corrupted versions:
six corruption types at three severities. Only images are altered. Ground-truth
label pixels are preserved and saved as PNG.

| Corruption ID | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- |
| `gaussian_noise` | σ=5 | σ=10 | σ=20 |
| `brightness_shift` | α=0.80, γ=1.10 | α=0.65, γ=1.30 | α=0.50, γ=1.60 |
| `uneven_light` | strength 0.35 | strength 0.60 | strength 0.85 |
| `motion_blur` | kernel 5 | kernel 9 | kernel 13 |
| `jpeg_compression` | quality 70 | quality 50 | quality 30 |
| `marker_distractor` | mask-aware, severity 1 | mask-aware, severity 2 | mask-aware, severity 3 |

Brightness shift applies `255 * clip(alpha * I / 255, 0, 1)^gamma` to each RGB
channel, then clips to [0, 255] and converts to uint8. Marker distractors
use ground-truth masks during data generation: severities 1 and 2 protect the
dilated crack region; severity 3 permits rare crossings under the generator's
sampling rules. The segmentation model receives only the resulting RGB image,
never the ground-truth mask as input. The detailed marker sampling parameters
are defined in `add_marker_distractor` in `generate_corruptions.py`.

## Generation

```shell
python generate_corruptions.py --root Steelcrack --out Steelcrack-C --seed 2026
```

The generator processes all six types in the order shown above. For base seed
b, zero-based corruption index c and severity s, the group seed is
`b + 1000*s + 7919*(c+1)`. The default base seed is 2026. Images are processed
in sorted filename order using NumPy's `default_rng`.

Each output group has `images/` and `masks/` under
`Steelcrack-C/<corruption>/<severity>/`. The generator records the input file
hashes, software versions, parameters, group seeds and output hashes in
`corruption_manifest.json`.

Rerunning the same command reproduces the complete random sequence, verifies
existing pixels and recreates missing files. `--overwrite` regenerates files
only in a directory with a matching manifest. Changed input files or settings
require a new output directory. An existing Steelcrack-C dataset in this layout
can also be used directly for evaluation.

## Evaluation

```shell
python evaluate_robustness.py --config configs/robustness.yaml
```

The selected validation-best checkpoint is used for all 19 test sets. The
configuration defaults to RGB ImageNet normalization, FP32, batch size 1 and
foreground probability greater than 0.5. To evaluate selected conditions:

```shell
python evaluate_robustness.py --config configs/robustness.yaml --datasets clean gaussian_noise/3
```

`--save-visualizations` enables lossless masks and original/GT/prediction panels.
`--no-benchmark` skips speed timing. Repeating a completed evaluation uses cached
tasks only when content hashes and outputs match; `--force` recomputes them.

## Results

| File | Contents |
| --- | --- |
| `summary.csv` | Per-model, per-dataset metrics as fractions, plus efficiency values |
| `summary_percent.csv` | Same quality metrics expressed as percentages |
| `robustness_summary.csv` | Mean corrupted performance, clean-relative drop and retention |
| `model_profile.csv` | One complexity/speed profile per model |
| `tasks/.../per_image.csv` | Per-image foreground metrics |
| `run_manifest.json` | Configuration, software/hardware and evaluation settings |

Dice, IoU, precision and recall use denominator smoothing of 1e-6. clDice uses
scikit-image skeletonization and the same smoothing. Each metric is averaged
over images; dataset-level confusion counts are not pooled. For empty prediction
and empty GT, Dice/IoU/precision/recall are 0 and clDice is 1. The default benchmark
requires positive ground-truth masks.

For a quality metric Q, the corrupted score is the arithmetic mean over the
18 conditions. The drop in percentage points is
`100 * (Q_clean - Q_corrupted)`, and retention is
`100 * Q_corrupted / Q_clean`. Per-corruption scores average its three severities.
Incomplete groups remain marked incomplete; missing values do not become zero.

Efficiency is measured once per model with a resident synthetic input and reused
in all dataset rows. FPS is images/s; latency is ms/image, with
`FPS = 1000 / latency`. It excludes data loading, CPU-to-GPU transfers, metric
calculation and visualization. It is not training throughput. Record the
PyTorch/CUDA/cuDNN versions and use the same environment for speed comparisons.

## Figures

```shell
python plot_robustness.py --input outputs/robustness/summary.csv --metric mi_iou
python plot_robustness.py --input outputs/robustness/summary.csv --metric mi_dice
```

The heatmap displays absolute performance in all 19 conditions. Severity curves
show how performance changes from the clean set to each corruption level. The
radar chart shows the average retained performance in each of the six corruption
categories; `--radar-metric performance` instead plots absolute corrupted scores.
All figures are saved as PNG, SVG and PDF. The plotting interface also accepts
standardized summaries containing multiple models for comparative figures.
