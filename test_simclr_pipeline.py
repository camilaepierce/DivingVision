#!/usr/bin/env python3
"""Test SimCLR augmentation and contrastive loss after tensor shape fix."""

import sys
import os
from pathlib import Path

# Change to the root directory
os.chdir(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.config import DivingConfig
from src.dataloader import create_loaders
from models.heavy_smart.model import build_model, augment_video_simclr

def test_simclr_pipeline():
    """Test that SimCLR augmentation and contrastive loss work."""
    print("=" * 80)
    print("Testing SimCLR Pipeline with Correct Tensor Shapes")
    print("=" * 80)
    
    # Setup
    cfg = DivingConfig("config.json")
    device = torch.device("cpu")
    model = build_model(cfg)
    model.to(device)
    
    # Get a batch
    print("\n1. Loading batch from dataloader...")
    train_loader, _ = create_loaders(cfg.getDataConfig(), batch_size=4, num_workers=0)
    
    for batch_videos, batch_labels in train_loader:
        print(f"   Input shape: {batch_videos.shape}")
        batch_videos = batch_videos.to(device)
        batch_labels = batch_labels.to(device)
        
        # Test augmentation
        print("\n2. Testing SimCLR augmentation...")
        videos_aug1 = augment_video_simclr(batch_videos, "moderate")
        videos_aug2 = augment_video_simclr(batch_videos, "moderate")
        print(f"   Augmented view 1 shape: {videos_aug1.shape}")
        print(f"   Augmented view 2 shape: {videos_aug2.shape}")
        # Note: Shape may change due to spatial cropping (224x224 -> 200x200)
        # but batch and channel dims should remain constant
        assert videos_aug1.ndim == 5
        assert videos_aug1.shape[0] == batch_videos.shape[0]  # Batch size
        assert videos_aug1.shape[2] == 3  # Channels
        assert videos_aug2.ndim == 5
        assert videos_aug2.shape[0] == batch_videos.shape[0]  # Batch size
        assert videos_aug2.shape[2] == 3  # Channels
        print("   ✓ Augmentation shapes correct!")
        
        # Test contrastive loss
        print("\n3. Testing contrastive loss...")
        try:
            loss = model.compute_contrastive_loss(videos_aug1, videos_aug2)
            print(f"   Contrastive loss: {loss.item():.4f}")
            print("   ✓ Contrastive loss computed successfully!")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            return False
        
        # Test supervised loss
        print("\n4. Testing supervised loss...")
        try:
            logits = model(batch_videos)
            sup_loss = model.compute_supervised_loss(logits, batch_labels)
            print(f"   Supervised loss: {sup_loss.item():.4f}")
            print("   ✓ Supervised loss computed successfully!")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            return False
        
        # Test combined loss
        print("\n5. Testing combined loss...")
        try:
            losses = model.compute_combined_loss(batch_videos, videos_aug1, batch_labels)
            print(f"   Total loss: {losses['total'].item():.4f}")
            print(f"   Supervised loss: {losses['supervised'].item():.4f}")
            print(f"   Contrastive loss: {losses['contrastive'].item():.4f}")
            print("   ✓ Combined loss computed successfully!")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
            return False
        
        break  # Only test first batch
    
    print("\n" + "=" * 80)
    print("All SimCLR tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_simclr_pipeline()
    sys.exit(0 if success else 1)
