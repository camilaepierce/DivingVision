#!/usr/bin/env python3
"""Quick test to verify tensor shapes after the Conv3d fix."""

import sys
import os
from pathlib import Path

# Change to the root directory to handle relative paths correctly
os.chdir(str(Path(__file__).parent))

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.config import DivingConfig
from src.dataloader import create_loaders
from models.heavy_smart.model import build_model

def test_tensor_shapes():
    """Test that tensor shapes are correct throughout the pipeline."""
    print("=" * 80)
    print("Testing tensor shapes after Conv3d fix")
    print("=" * 80)
    
    # Load config and create dataloaders
    print("\n1. Creating data loaders...")
    cfg = DivingConfig("config.json")
    train_loader, _ = create_loaders(cfg.getDataConfig(), batch_size=2, num_workers=0)
    
    # Get a single batch
    print("2. Getting a batch from dataloader...")
    for batch_videos, batch_labels in train_loader:
        print(f"   Batch video shape: {batch_videos.shape}")
        print(f"   Expected: (B, T, C, H, W) = (B, 16, 3, 224, 224)")
        
        # Verify shape
        assert batch_videos.ndim == 5, f"Expected 5D tensor, got {batch_videos.ndim}D"
        B, T, C, H, W = batch_videos.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert H == 224 and W == 224, f"Expected 224x224, got {H}x{W}"
        print("   ✓ Batch shape is correct!")
        
        # Test with model
        print("\n3. Testing with model...")
        device = torch.device("cpu")
        model = build_model(cfg)
        model.to(device)
        
        batch_videos = batch_videos.to(device)
        
        print("4. Running forward pass...")
        try:
            logits = model(batch_videos)
            print(f"   Model output shape: {logits.shape}")
            print(f"   Expected: (B, num_classes) = (2, 48)")
            print("   ✓ Forward pass successful!")
        except RuntimeError as e:
            print(f"   ✗ ERROR: {e}")
            return False
        
        break  # Only test first batch
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_tensor_shapes()
    sys.exit(0 if success else 1)
