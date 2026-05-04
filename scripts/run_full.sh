#!/bin/bash
#SBATCH -J diving48_resnet50
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /home/jacktuck/DivingVision/logs/full_%j.out
#SBATCH -e /home/jacktuck/DivingVision/logs/full_%j.err

set -e

mkdir -p /home/jacktuck/DivingVision/logs
cd /home/jacktuck/DivingVision

module load miniforge
conda activate diving48_baselines

python train_diving48_resnet50.py \
  --run-name full_resnet50_16f_meanpool \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 8 \
  --lr 1e-4 \
  --weight-decay 1e-4
