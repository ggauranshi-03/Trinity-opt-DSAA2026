#!/bin/bash
# Move to repository root regardless of where script is called from
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# GPU 3 is currently returning an Unknown Error, so we schedule across GPUs 0, 1, and 2.
echo "Starting FO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -u train_transformer_dsaa.py --variant FO --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_FO.log 2>&1 &
PID0=$!

echo "Starting ZO on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -u train_transformer_dsaa.py --variant ZO --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_ZO.log 2>&1 &
PID1=$!

echo "Starting SO on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python3 -u train_transformer_dsaa.py --variant SO --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_SO.log 2>&1 &
PID2=$!

# Wait for the first batch to finish
wait $PID0
echo "FO finished. Starting FO_ZO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -u train_transformer_dsaa.py --variant FO_ZO --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_FO_ZO.log 2>&1 &

wait $PID1
echo "ZO finished. Starting FO_SO on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -u train_transformer_dsaa.py --variant FO_SO --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_FO_SO.log 2>&1 &

wait $PID2
echo "SO finished. Starting Full on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python3 -u train_transformer_dsaa.py --variant Full --epochs 100 --dataset cifar10 --imb_factor 0.01 > logs/transformer_Full.log 2>&1 &

# Wait for the second batch to finish
wait
echo "All Transformer ablation studies (100 epochs) have completed!"
