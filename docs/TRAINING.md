# Training protocol

`configs/hg_crackseg.yaml` is the complete default configuration. Structural
ablations use the same training and loss settings.

| Setting | Value |
| --- | --- |
| Input | RGB, 512 × 512 |
| Normalization | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Epochs / batch size | 100 / 8 |
| Optimizer | AdamW, default betas (0.9, 0.999), epsilon 1e-8 |
| Base learning rate / weight decay | 0.0003 / 0.0001 |
| Scheduler | CosineAnnealingLR, T_max=100, eta_min=1e-6 |
| Hessian learning rate | 0.008 × base initial learning rate |
| Hessian freeze | Epochs 1–5 inclusive |
| Gradient clipping | Global norm 1.0 |
| Seed | 42 |
| Selection metric | Highest validation mean per-image Dice |

The optimizer assigns all Hessian parameters to a separate parameter group.
Cosine annealing applies to both groups with the configured common minimum
learning rate; their learning-rate ratio therefore changes over training.
Freezing affects the entire Hessian module during the initial five epochs.

Augmentation consists of horizontal flip (p=0.5), vertical flip (p=0.5), random
90-degree rotation (p=0.5), and grid distortion (5 steps, limit 0.3, p=0.3),
followed by normalization and tensor conversion. Validation uses normalization
and tensor conversion only. Images are not resized by these transforms.

The head emits background/crack logits. For crack probability p, the loss uses
q = sigmoid(50 × (p − 0.5)) and combines unweighted cross entropy with foreground
Dice loss on q, with equal weights and Dice smoothing 1e-6. All supplied structural
ablation configurations use the same loss.

## Checkpoints and logs

`best_model.pth` contains only the validation-selected state dictionary.
`last_checkpoint.pth` contains the model, best model, optimizer, scheduler,
epoch, history, random-number states and data hashes. Periodic state dictionaries
are saved every five epochs. `history.csv` records train/validation losses,
image-wise metrics, actual group learning rates and epoch duration.

```shell
python train.py --config configs/hg_crackseg.yaml --resume checkpoints/hg_crackseg/last_checkpoint.pth
```

Resume restores an epoch boundary. It requires unchanged data and recipe,
matching libraries and numerical runtime settings, and `num_workers: 0`. Interrupted
partial epochs are rerun. `training.init_weight` starts a new run from strictly
loaded weights; it does not restore optimizer state. Test data is never used
for training or checkpoint selection.

For a short execution check, `--epochs 1` stops after the first epoch while
retaining the YAML scheduler horizon. Use a separate output directory:

```shell
python train.py --config configs/hg_crackseg.yaml --epochs 1 --output-dir outputs/train_check
```

For CPU execution, pass `--device cpu`. The default model contains BatchNorm layers, so training
requires at least two images per batch, including the final batch.

W&B is disabled by default. To enable it, install `wandb` separately and set
`training.wandb_mode` to `online` or `offline`; local logs remain available.

## Checkpoint loading

`core/checkpoint.py` accepts a state dictionary or a dictionary containing
`state_dict`, `model_state_dict` or `model`. It removes an optional `module.`
prefix and maps supported earlier parameter names before strict loading.
Duplicate keys, missing or unexpected parameters, and shape mismatches raise
an error. Use the same architecture configuration for training and evaluation.
