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
├── trinity_optimizer.py               # Standalone Trinity optimizer (ZO sensor + gated K-FAC + AdamW)
├── datasets_lt.py                     # Long-tailed CIFAR-10/100 datasets & shot evaluators
├── config.yaml                        # ALL hyperparameters: dataset, training, wandb, per-optimizer
├── optimizer_factory.py               # Builds Adam/RMSprop/SGD/Trinity + scheduler from config.yaml
│
├── train_resnet18_dsaa.py             # ResNet-18 benchmark training script (--optimizer flag)
├── train_wide_resnet_dsaa.py          # Wide-ResNet-101-2 benchmark training script (--optimizer flag)
├── train_transformer_dsaa.py          # 100M Transformer (ViT) benchmark training script (--optimizer flag)
├── plot_optimizer_comparison.py       # Accuracy-vs-epoch comparison plot across optimizers
│
├── run_full_ablation.py               # Ablation study runner (FO, ZO, SO, FO_ZO, FO_SO, Full)
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

Import and use `Trinity` just like any standard PyTorch optimizer. `step()` requires a
`closure()` that recomputes the loss under `torch.no_grad()` for the same minibatch —
it's needed by the forward-only ZO curvature sensor:

```python
import torch
import torchvision.models as models
from trinity_optimizer import Trinity

model = models.resnet18(num_classes=10).cuda()
optimizer = Trinity(
    model=model,
    lr=1e-3,
    damping=1e-2,           # K-FAC damping factor
    sensor_interval=100,    # ZO curvature sensing interval (steps)
    sensor_probes=4,        # antithetic probes per sensed block
    beta_max=0.3,           # fraction of blocks allowed in K-FAC (SO) mode
    use_so=True,            # enable gated K-FAC preconditioning
    use_zo=True,            # enable ZO sensing + saddle-escape actuator
)

criterion = torch.nn.CrossEntropyLoss()

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    def closure():
        with torch.no_grad():
            return criterion(model(inputs), targets)

    optimizer.step(closure=closure)
```

All hyperparameters used in the benchmark scripts below (including the full Trinity
sensor/controller/K-FAC/escape configuration) live in [`config.yaml`](config.yaml) —
nothing is hardcoded in the training scripts.

---

## 🏃 Running Experiments

### Benchmarks: comparing optimizers

Each training script accepts `--optimizer {adam,rmsprop,sgd,trinity}` and reads every
other setting (dataset, batch size, epochs, LR schedule, per-optimizer hyperparameters)
from [`config.yaml`](config.yaml), so all four optimizers are trained under identical
conditions — same architecture, same Long-Tailed CIFAR data, same schedule.

Edit `config.yaml` to change the dataset/imbalance factor/epoch budget, then run all
four optimizers for a given architecture:

```bash
# ResNet-18
python train_resnet18_dsaa.py --optimizer adam
python train_resnet18_dsaa.py --optimizer rmsprop
python train_resnet18_dsaa.py --optimizer sgd
python train_resnet18_dsaa.py --optimizer trinity

# Wide-ResNet-101-2
python train_wide_resnet_dsaa.py --optimizer adam
python train_wide_resnet_dsaa.py --optimizer rmsprop
python train_wide_resnet_dsaa.py --optimizer sgd
python train_wide_resnet_dsaa.py --optimizer trinity

# 100M Transformer (ViT-style)
python train_transformer_dsaa.py --optimizer adam
python train_transformer_dsaa.py --optimizer rmsprop
python train_transformer_dsaa.py --optimizer sgd
python train_transformer_dsaa.py --optimizer trinity
```

Or loop over all three architectures and four optimizers in one go:

```bash
for opt in adam rmsprop sgd trinity; do
  python train_resnet18_dsaa.py --optimizer $opt
  python train_wide_resnet_dsaa.py --optimizer $opt
  python train_transformer_dsaa.py --optimizer $opt
done
```

Each run writes `results_dsaa2026/{model}_{optimizer}_{dataset}_imb{imb_factor}.csv`
with per-epoch train/val loss, accuracy, and head/medium/tail shot accuracy.

**wandb.** Every run also logs to Weights & Biases. The project name comes from
`wandb.project` in `config.yaml` (default `trinity-optimizer-comparison`), suffixed
per architecture — e.g. `trinity-optimizer-comparison-resnet18` — so **one wandb
project holds every optimizer's run for a given model + dataset** as separate,
directly comparable runs (named `adam`, `rmsprop`, `sgd`, `trinity`).

### Comparison plots

Once the CSVs for all four optimizers exist for a model, generate the
accuracy-vs-epoch comparison plot (converging curves with final-accuracy labels):

```bash
python plot_optimizer_comparison.py --model resnet18
python plot_optimizer_comparison.py --model wideresnet
python plot_optimizer_comparison.py --model transformer
# or all three at once:
python plot_optimizer_comparison.py --model all
```

Plots are saved to `figs/{model}_{dataset}_imb{imb_factor}_val_acc.png`.

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
