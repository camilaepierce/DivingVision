#!/bin/bash
#SBATCH -J diving48_smoke_aug
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /home/jacktuck/DivingVision/logs/smoke_aug_%j.out
#SBATCH -e /home/jacktuck/DivingVision/logs/smoke_aug_%j.err

set -e

mkdir -p /home/jacktuck/DivingVision/logs
cd /home/jacktuck/DivingVision

module load miniforge
conda activate diving48_baselines

python train_diving48_resnet50_aug.py \
  --run-name smoke_resnet50_16f_meanpool_aug_dropout \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 4 \
  --max-train-samples 128 \
  --max-test-samples 64 \
  --limit-train-batches 10