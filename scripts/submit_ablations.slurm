#!/bin/bash
#SBATCH --partition=gpu               # Send to the GPU nodes
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:2                  # Request 4 GPUs on that node
#SBATCH --time=04:00:00               # Max time (4 hours is plenty)
#SBATCH --job-name=trinity_ablation
#SBATCH --output=logs/slurm_%j.log   # Master log file in logs directory

# Load your environment (You may need to change 'fls' to whatever your conda env is named on this cluster)
# module load miniconda3
# source activate fls

# Ensure logs directory exists
mkdir -p logs

# Run the transformer ablation script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_transformer_ablations.sh"

