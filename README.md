# Trinity: Tri-Level Hybrid Optimization for Deep Long-Tailed Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/Conference-DSAA%202026-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official research repository for **Trinity**, a tri-level hybrid optimization framework designed for training deep neural networks under extreme class imbalance and long-tailed data distributions.

---

## 🔬 Core Methodology

The **Trinity** optimizer synergistically integrates three complementary optimization paradigms into a unified update rule:

1. **First-Order (FO) Momentum**: Employs adaptive first and second moment estimations (AdamW-style) for steady, low-variance parameter descent.
2. **Second-Order (SO) Curvature Preconditioning**: Computes Kronecker-factored Approximate Curvature (**K-FAC**) on linear and convolutional layers, capturing second-order loss landscape geometry while maintaining linear per-step overhead.
3. **Zeroth-Order (ZO) Gradient Augmentation**: Periodically executes finite-difference stochastic gradient estimation along random orthogonal directions, enabling parameters to escape local minima and saddle points often encountered in scarce tail classes.

```text
                  ┌────────────────────────────────────────┐
                  │            Trinity Optimizer           │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  [ First-Order (FO) ]       [ Second-Order (SO) ]        [ Zeroth-Order (ZO) ]
  Adaptive AdamW updates     K-FAC Preconditioning        Stochastic Exploration
  (Gradient Moments)         (Loss Landscape Curvature)   (Escaping Saddle Points)
```

---

## 📁 Repository Structure

The repository is neatly structured and organized for seamless reproducibility:

```text
Trinity-opt-DSAA2026/
├── README.md                          # Main project documentation & reproduction guide
├── requirements.txt                   # Environment dependencies
├── .gitignore                         # Git exclusion rules
│
├── trinity_optimizer.py               # Standalone Trinity optimizer (FO + SO + ZO)
├── datasets_lt.py                     # Long-tailed CIFAR-10/100 datasets & shot evaluators
│
├── train_resnet18_dsaa.py             # ResNet-18 benchmark training script
├── train_wide_resnet_dsaa.py          # Wide-ResNet-101-2 benchmark training script
├── train_transformer_dsaa.py          # 100M Transformer (ViT) benchmark training script
│
├── run_full_ablation.py               # Ablation study runner (FO, ZO, SO, FO_ZO, FO_SO, Full)
├── run_sensitivity_and_optuna.py      # Optuna hyperparameter sweeps & sensitivity analysis
│
├── scripts/                           # Execution bash & SLURM scripts
│   ├── README.md                      # Documentation of execution scripts
│   ├── run_all.sh                     # Concurrent execution of benchmarks across GPUs
│   ├── run_all_ablations.sh           # CIFAR-10 ablation suite
│   ├── run_all_ablations_cifar100.sh  # CIFAR-100 extreme imbalance ablation suite
│   ├── run_transformer_ablations.sh   # Transformer ablation suite
│   └── submit_ablations.slurm         # SLURM cluster submission file
│
├── logs/                              # Training and ablation run execution logs
│   ├── README.md                      # Log taxonomy & naming index
│   ├── ablation_*.log                 # CIFAR-10 & CIFAR-100 ablation logs
│   ├── transformer*.log               # Transformer training logs
│   └── wideresnet_run.log             # Wide-ResNet run logs
│
├── results_dsaa2026/                  # Official DSAA 2026 benchmark and ablation CSV outputs
│   ├── README.md                      # Metrics schema & CSV format documentation
│   └── *.csv                          # Per-epoch accuracy and loss records
│
└── extra/                             # Archived legacy prototypes and raw cluster logs
    ├── README.md                      # Index and explanation of archived resources
    ├── legacy_scripts/                # Early monolithic prototype scripts
    ├── baseline_experiments/          # Preliminary experiment runs (exp1, exp2, exp3)
    └── slurm_master_logs/             # Cluster stdout/stderr dumps from batch jobs
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ggauranshi-03/Trinity-opt-DSAA2026.git
cd Trinity-opt-DSAA2026
pip install -r requirements.txt
```

### 2. Using the Trinity Optimizer

Import and use `Trinity` just like any standard PyTorch optimizer:

```python
import torch
import torchvision.models as models
from trinity_optimizer import Trinity

model = models.resnet18(num_classes=10).cuda()
optimizer = Trinity(
    model=model,
    lr=1e-3,
    damping=1e-2,       # K-FAC damping factor
    kfac_interval=10,   # K-FAC curvature update interval
    zo_interval=20,     # Zeroth-order perturbation interval
    zo_scale=0.1,       # Zeroth-order augmentation scale
    use_fo=True,
    use_so=True,
    use_zo=True
)

criterion = torch.nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Define loss closure for Zeroth-Order finite-difference steps
    def closure():
        with torch.no_grad():
            return criterion(model(inputs), targets)

    optimizer.step(closure=closure)
```

---

## 🏃 Running Experiments

### Benchmarks

Run benchmarks individually across architectures:

```bash
# Train ResNet-18 on Long-Tailed CIFAR-10 (imbalance 0.01, 300 epochs)
python3 train_resnet18_dsaa.py --dataset cifar10 --imb_factor 0.01 --epochs 300

# Train Wide-ResNet-101-2 on Long-Tailed CIFAR-10 (imbalance 0.01, 200 epochs)
python3 train_wide_resnet_dsaa.py --dataset cifar10 --imb_factor 0.01 --epochs 200

# Train 100M Transformer on Long-Tailed CIFAR-10 (100 epochs)
python3 train_transformer_dsaa.py --dataset cifar10 --imb_factor 0.01 --epochs 100
```

### Ablation Studies

Evaluate component contributions (`FO`, `ZO`, `SO`, `FO_ZO`, `FO_SO`, `Full`):

```bash
python3 run_full_ablation.py --variant Full --dataset cifar10 --imb_factor 0.01 --epochs 100
python3 run_full_ablation.py --variant FO   --dataset cifar10 --imb_factor 0.01 --epochs 100
python3 run_full_ablation.py --variant SO   --dataset cifar10 --imb_factor 0.01 --epochs 100
python3 run_full_ablation.py --variant ZO   --dataset cifar10 --imb_factor 0.01 --epochs 100
```

### Automated Batch Runs

Batch scripts in `scripts/` handle multi-GPU job orchestration:

```bash
# Run all benchmark models in parallel across GPUs 1, 2, 3
bash scripts/run_all.sh

# Run the complete CIFAR-10 ablation suite across 4 GPUs
bash scripts/run_all_ablations.sh

# Run CIFAR-100 extreme imbalance (0.005) ablation suite
bash scripts/run_all_ablations_cifar100.sh

# Submit batch ablation job to a SLURM cluster
sbatch scripts/submit_ablations.slurm
```

---

## 📊 Evaluation & Metrics

Evaluation scripts compute overall validation metrics alongside stratified accuracy across class frequencies:
- **Many-Shot (`head_acc`)**: Classes with $>100$ training samples.
- **Medium-Shot (`med_acc`)**: Classes with $20$ to $100$ training samples.
- **Few-Shot (`tail_acc`)**: Rare tail classes with $<20$ training samples.

Metric logs are saved directly to [`results_dsaa2026/`](results_dsaa2026/).

---

## 📦 Archived Resources

For historical reference, exploratory experiments (`exp1`, `exp2`, `exp3`), monolithic prototype scripts, and raw cluster output dumps are cataloged in [`extra/`](extra/).
