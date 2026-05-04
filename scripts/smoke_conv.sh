#!/bin/bash
#SBATCH -J diving48_smoke_tconv
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /home/jacktuck/DivingVision/logs/smoke_tconv_%j.out
#SBATCH -e /home/jacktuck/DivingVision/logs/smoke_tconv_%j.err

set -e

mkdir -p /home/jacktuck/DivingVision/logs
cd /home/jacktuck/DivingVision

module load miniforge
conda activate diving48_baselines

python train_diving48_resnet50_temporal_conv.py \
  --run-name smoke_resnet50_16f_temporal_conv \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 4 \
  --max-train-samples 128 \
  --max-test-samples 64 \
  --limit-train-batches 10