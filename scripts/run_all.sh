#!/bin/bash
# Move to repository root regardless of where script is called from
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

echo "Starting Transformer on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 train_transformer_dsaa.py > logs/transformer.log 2>&1 &

echo "Waiting 30 seconds..."
sleep 30

echo "Starting Wide-ResNet on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python3 train_wide_resnet_dsaa.py > logs/wideresnet.log 2>&1 &

echo "Waiting 30 seconds..."
sleep 30

echo "Starting Ablation on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python3 run_full_ablation.py > logs/ablation.log 2>&1 &

echo "Waiting for all background jobs to finish..."
wait
echo "All done!"
