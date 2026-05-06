#!/usr/bin/env python3
"""Short smoke test training run for the heavy model."""

import json
import torch
from torch.utils.data import Subset
from src.config import DivingConfig
from src.dataloader import create_loaders
from src.training import train_model
from models.heavy.model import build_model


def main():
    # Load config
    cfg = DivingConfig("config.json")
    data_cfg = cfg.getDataConfig()
    
    # Fix paths to absolute
    import os
    base_dir = "/home/camila/CompVis/DivingVision"
    data_cfg["rgb_data"] = os.path.join(base_dir, "data/Diving48_rgb")
    data_cfg["train_split"] = os.path.join(base_dir, "data/Diving48_V2_train.json")
    data_cfg["test_split"] = os.path.join(base_dir, "data/Diving48_V2_test.json")
    
    # Create full loaders
    train_loader, test_loader = create_loaders(data_cfg, batch_size=8, num_workers=2)
    
    # Limit to first 500 training samples for smoke test
    train_dataset = train_loader.dataset
    if len(train_dataset) > 500:
        train_dataset = Subset(train_dataset, range(500))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # No workers on CPU
            pin_memory=False
        )
    
    print(f"Training on {len(train_dataset)} samples for 3 epochs")
    print(f"Test set: {len(test_loader.dataset)} samples (limited to first 100)")
    
    # Limit test set to first 100 samples for faster evaluation
    test_dataset = test_loader.dataset
    if len(test_dataset) > 100:
        test_dataset = Subset(test_dataset, range(100))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    # Build model
    model = build_model(cfg)
    print(f"Model built: {model.__class__.__name__}")
    
    # Train
    train_model(
        model,
        train_loader,
        test_loader,
        device=None,  # Auto-detect
        num_epochs=3,
        learning_rate=0.01,
        use_mixed_precision=True,
        eval_every=2,
        max_eval_batches=20,
        visualize_history=False
    )
    
    print("\nSmoke test completed successfully!")


if __name__ == "__main__":
    main()
