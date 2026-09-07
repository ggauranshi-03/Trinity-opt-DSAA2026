# Execution Scripts

This directory contains shell scripts and cluster job submission files for running benchmarks and ablation experiments.

## Scripts Overview

All scripts automatically resolve the workspace root directory and ensure that logs are written to the `logs/` directory.

| Script | Description | Recommended Execution |
| :--- | :--- | :--- |
| `run_all.sh` | Launches Transformer (GPU 1), Wide-ResNet (GPU 2), and Ablation (GPU 3) concurrently. | `bash scripts/run_all.sh` |
| `run_all_ablations.sh` | Runs 100-epoch ablation suite on CIFAR-10 across 4 GPUs (`FO`, `ZO`, `SO`, `FO_ZO`, `FO_SO`, `Full`). | `bash scripts/run_all_ablations.sh` |
| `run_all_ablations_cifar100.sh` | Runs 100-epoch ablation suite on CIFAR-100 (extreme imbalance `0.005`) across 4 GPUs. | `bash scripts/run_all_ablations_cifar100.sh` |
| `run_transformer_ablations.sh` | Schedules Transformer ablations on GPUs 0, 1, and 2. | `bash scripts/run_transformer_ablations.sh` |
| `submit_ablations.slurm` | SLURM batch job script to submit Transformer ablation suite to cluster GPU queue. | `sbatch scripts/submit_ablations.slurm` |

## Output Redirection
Logs from these scripts are placed in `logs/` and experimental metrics are saved into `results_dsaa2026/`.
