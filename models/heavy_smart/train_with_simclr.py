#!/usr/bin/env python3
"""
Example training script demonstrating SimCLR contrastive learning with heavy_smart model.

This script shows three training approaches:
1. Pure contrastive pre-training (self-supervised)
2. Supervised fine-tuning after pre-training
3. Combined multi-task learning

Usage:
    python train_with_simclr.py --mode pretrain --epochs 50 --device cuda
    python train_with_simclr.py --mode finetune --epochs 20 --device cuda
    python train_with_simclr.py --mode combined --epochs 40 --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.heavy_smart.model import build_model, augment_video_simclr
from models.heavy_smart.simclr_training import (
    train_epoch_contrastive,
    train_epoch_supervised,
    train_epoch_combined,
    evaluate_supervised,
)
from src.config import DivingConfig
from src.dataloader import create_loaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train heavy_smart model with SimCLR contrastive learning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune", "combined"],
        default="combined",
        help="Training mode: pretrain (contrastive), finetune (supervised), or combined"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/simclr",
        help="Directory to save model checkpoints and logs"
    )
    parser.add_argument(
        "--augmentation_level",
        type=str,
        choices=["light", "moderate", "strong"],
        default="moderate",
        help="Augmentation intensity level"
    )
    parser.add_argument(
        "--supervised_weight",
        type=float,
        default=1.0,
        help="Weight for supervised loss in combined mode"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Weight for contrastive loss in combined mode"
    )
    return parser.parse_args()


def create_checkpoint_path(save_dir: str, mode: str, epoch: int) -> str:
    """Create checkpoint filename."""
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"heavy_smart_{mode}_epoch_{epoch:03d}.pt")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int,
                    save_path: str, mode: str, stats: dict):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'mode': mode,
        'stats': stats,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                    checkpoint_path: str) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"Training heavy_smart model with SimCLR")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Augmentation: {args.augmentation_level}")
    print()
    
    # Initialize model
    print("Initializing model...")
    cfg_obj = DivingConfig(filename="config.json")
    model = build_model(cfg_obj)
    model.to(device)
    print(f"Model created with SimCLR enabled: {model.enable_simclr}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader = create_loaders(
        cfg_obj.getDataConfig(),
        batch_size=args.batch_size,
        num_workers=4
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    start_epoch = 0
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
    
    # Training loop
    print(f"\nStarting training ({args.mode} mode)...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        
        if args.mode == "pretrain":
            # Pure contrastive pre-training
            stats = train_epoch_contrastive(
                model, train_loader, optimizer, device,
                augmentation_level=args.augmentation_level
            )
            print(f"Epoch {epoch_num}/{args.epochs}: "
                  f"Contrastive Loss = {stats['contrastive_loss']:.4f}")
        
        elif args.mode == "finetune":
            # Supervised fine-tuning
            sup_stats = train_epoch_supervised(model, train_loader, optimizer, device)
            val_stats = evaluate_supervised(model, test_loader, device)
            print(f"Epoch {epoch_num}/{args.epochs}: "
                  f"Train Loss = {sup_stats['loss']:.4f}, "
                  f"Val Loss = {val_stats['loss']:.4f}, "
                  f"Val Acc = {val_stats['accuracy']:.4f}")
        
        elif args.mode == "combined":
            # Combined multi-task learning
            stats = train_epoch_combined(
                model, train_loader, optimizer, device,
                supervised_weight=args.supervised_weight,
                contrastive_weight=args.contrastive_weight,
                augmentation_level=args.augmentation_level
            )
            print(f"Epoch {epoch_num}/{args.epochs}: "
                  f"Total Loss = {stats['total_loss']:.4f}, "
                  f"Sup Loss = {stats['supervised_loss']:.4f}, "
                  f"Cont Loss = {stats['contrastive_loss']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch_num % 10 == 0) or (epoch_num == args.epochs):
            checkpoint_path = create_checkpoint_path(args.save_dir, args.mode, epoch_num)
            save_checkpoint(model, optimizer, epoch, checkpoint_path, args.mode, stats)
    
    print("=" * 80)
    print("Training completed!")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f"heavy_smart_{args.mode}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
