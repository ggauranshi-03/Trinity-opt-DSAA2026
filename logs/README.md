# Training & Ablation Logs

This directory contains consolidated stdout logs from model training and ablation runs.

## Log Categories

### 1. Ablation Study Logs
- `ablation_<variant>.log`: CIFAR-10 ablation runs (100 epochs, imbalance factor 0.01).
- `ablation_<variant>_cifar100.log`: CIFAR-100 ablation runs (100 epochs, imbalance factor 0.01).
- `ablation_<variant>_cifar100_ext.log`: CIFAR-100 extreme imbalance ablation runs (100 epochs, imbalance factor 0.005).

Variants:
- `FO`: First-Order only (AdamW baseline)
- `ZO`: Zeroth-Order only (gradient-free random perturbations)
- `SO`: Second-Order only (K-FAC curvature)
- `FO_ZO`: First-Order + Zeroth-Order hybrid
- `FO_SO`: First-Order + Second-Order hybrid
- `Full`: Complete Trinity optimizer (FO + SO + ZO)

### 2. Model Benchmark Logs
- `wideresnet_run.log`: Wide-ResNet-101-2 benchmark run (200 epochs).
- `transformer_run.log`: 100M Transformer benchmark run.
- `transformer_<variant>.log`: Transformer ablation study logs across Trinity components.
