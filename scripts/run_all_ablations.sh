#!/bin/bash
# Move to repository root regardless of where script is called from
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# Run first batch of 4 variants on the 4 GPUs
echo "Starting FO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -u run_full_ablation.py --variant FO --epochs 100 > logs/ablation_FO.log 2>&1 &
PID0=$!

echo "Starting ZO on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -u run_full_ablation.py --variant ZO --epochs 100 > logs/ablation_ZO.log 2>&1 &
PID1=$!

echo "Starting SO on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python3 -u run_full_ablation.py --variant SO --epochs 100 > logs/ablation_SO.log 2>&1 &
PID2=$!

echo "Starting FO_ZO on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python3 -u run_full_ablation.py --variant FO_ZO --epochs 100 > logs/ablation_FO_ZO.log 2>&1 &
PID3=$!

# Wait for GPU 0 and GPU 1 to finish so we can reuse them
wait $PID0
echo "FO finished. Starting FO_SO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -u run_full_ablation.py --variant FO_SO --epochs 100 > logs/ablation_FO_SO.log 2>&1 &

wait $PID1
echo "ZO finished. Starting Full on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -u run_full_ablation.py --variant Full --epochs 100 > logs/ablation_Full.log 2>&1 &

# Wait for all remaining background processes to finish
wait
echo "All ablation studies (100 epochs) have completed!"
