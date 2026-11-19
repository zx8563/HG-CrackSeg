# Structural ablations

Each file in `configs/ablations/` is a complete configuration. All use the same
training protocol, probability-hardening loss and attention gates. Models with
the Hessian branch use two levels of geometric guidance.

| Experiment | Configuration | Hessian | Large-kernel blocks | ASPP |
| --- | --- | --- | --- | --- |
| M0 | `m0.yaml` | No | No | No |
| M1 | `m1.yaml` | No | Yes | Yes |
| M2 | `m2.yaml` | Yes | No | No |
| M3 | `m3.yaml` | Yes | Yes | Yes |
| M2 + large kernel | `m2_large_kernel.yaml` | Yes | Yes | No |
| M2 + ASPP | `m2_aspp.yaml` | Yes | No | Yes |

With ASPP disabled, a standard convolutional block occupies the bottleneck.
Disabling the Hessian branch removes its feature stream and adjusts the decoder
input channels. Disabling the large-kernel blocks replaces them with identities.

| Experiment | Configuration | Change from the full model |
| --- | --- | --- |
| Fixed Hessian kernels | `fixed_kernel.yaml` | Kernel residual scale 0 |
| Learnable Hessian kernels | `learnable_kernel.yaml` | Kernel residual scale 0.03 |
| Single scale σ=1 | `sigma_1.yaml` | Hessian sigmas [1] |
| Single scale σ=2 | `sigma_2.yaml` | Hessian sigmas [2] |
| Single scale σ=4 | `sigma_4.yaml` | Hessian sigmas [4] |
| Multi-scale stacking | `sigma_stack.yaml` | Hessian sigmas [1,2,4] |

The fixed-kernel experiment disables only the residual contribution to the
Hessian kernels. The other learnable Hessian response parameters follow the
same optimizer and freeze schedule. Multi-scale responses are concatenated
as separate feature channels.

`m3`, `learnable_kernel`, `sigma_stack` and the main `hg_crackseg` configuration
describe the same full model and recipe. Their separate run names identify the
corresponding experiment groups; one full-model run can supply their common
reference. A new run with the same seed is not an independent seed replicate.

```shell
python train.py --config configs/ablations/m0.yaml
python test.py --config configs/ablations/m0.yaml --checkpoint checkpoints/m0/best_model.pth --output-dir outputs/m0
```

Replace `m0` with the required experiment name. Select each experiment's
checkpoint using validation Dice before evaluating the test set.
