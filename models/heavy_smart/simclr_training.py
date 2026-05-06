"""
Training utilities for SimCLR contrastive learning with the heavy_smart model.

This module provides helper functions and classes for training with:
1. Pure contrastive learning (self-supervised pre-training)
2. Combined supervised + contrastive learning (multi-task learning)
3. Fine-tuning after contrastive pre-training
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def _tqdm(iterable, desc=None, **kwargs):
    """Wrapper around tqdm that falls back to iter if tqdm is not available."""
    if HAS_TQDM:
        return tqdm.tqdm(iterable, desc=desc, **kwargs)
    else:
        if desc:
            print(f"[{desc}] Processing...")
        return iter(iterable)


def train_epoch_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_fn = None
) -> Dict[str, float]:
    """Train one epoch with supervised learning.
    
    Args:
        model: Heavy model with SimCLR
        train_loader: DataLoader providing (video, label) pairs
        optimizer: Optimizer for model parameters
        device: Device to train on (cuda or cpu)
        loss_fn: Loss function (uses model's compute_supervised_loss if None)
        
    Returns:
        Dictionary with epoch statistics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for videos, labels in _tqdm(train_loader, desc="Training"):
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(videos)
        
        # Compute loss
        if loss_fn is None:
            loss = model.compute_supervised_loss(logits, labels)
        else:
            loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def train_epoch_contrastive(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    augmentation_level: str = "moderate"
) -> Dict[str, float]:
    """Train one epoch with pure contrastive learning (self-supervised).
    
    Args:
        model: Heavy model with SimCLR enabled
        train_loader: DataLoader providing video samples
        optimizer: Optimizer for model parameters
        device: Device to train on
        augmentation_level: "light", "moderate", or "strong"
        
    Returns:
        Dictionary with epoch statistics
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    from models.heavy_smart.model import augment_video_simclr
    
    for batch in _tqdm(train_loader, desc="Contrastive Training"):
        # Handle both (video, label) and (video,) formats
        if isinstance(batch, (list, tuple)):
            videos = batch[0]
        else:
            videos = batch
        
        videos = videos.to(device)
        
        # Create two augmented views
        videos_aug1 = augment_video_simclr(videos, augmentation_level)
        videos_aug2 = augment_video_simclr(videos, augmentation_level)
        
        optimizer.zero_grad()
        
        # Compute contrastive loss
        loss = model.compute_contrastive_loss(videos_aug1, videos_aug2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'contrastive_loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def train_epoch_combined(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    supervised_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    augmentation_level: str = "moderate"
) -> Dict[str, float]:
    """Train one epoch with combined supervised + contrastive learning.
    
    This approach enables multi-task learning where the model learns both
    from labeled data (supervised) and unlabeled augmented views (contrastive).
    
    Args:
        model: Heavy model with SimCLR enabled
        train_loader: DataLoader providing (video, label) pairs
        optimizer: Optimizer for model parameters
        device: Device to train on
        supervised_weight: Weight for supervised loss
        contrastive_weight: Weight for contrastive loss
        augmentation_level: "light", "moderate", or "strong"
        
    Returns:
        Dictionary with epoch statistics
    """
    model.train()
    total_sup_loss = 0.0
    total_cont_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    from models.heavy_smart.model import augment_video_simclr
    
    for videos, labels in _tqdm(train_loader, desc="Combined Training"):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Create augmented views
        videos_aug = augment_video_simclr(videos, augmentation_level)
        
        optimizer.zero_grad()
        
        # Compute combined loss
        loss_dict = model.compute_combined_loss(
            videos, videos_aug, labels,
            supervised_weight=supervised_weight,
            contrastive_weight=contrastive_weight
        )
        
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sup_loss += loss_dict['supervised'].item()
        total_cont_loss += loss_dict['contrastive'].item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'supervised_loss': total_sup_loss / num_batches,
        'contrastive_loss': total_cont_loss / num_batches,
        'num_batches': num_batches
    }


@torch.no_grad()
def evaluate_supervised(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    loss_fn = None
) -> Dict[str, float]:
    """Evaluate model with supervised metric (accuracy + loss).
    
    Args:
        model: Heavy model
        test_loader: DataLoader providing (video, label) pairs
        device: Device to evaluate on
        loss_fn: Loss function (uses model's compute_supervised_loss if None)
        
    Returns:
        Dictionary with evaluation statistics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for videos, labels in test_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(videos)
        
        # Compute loss
        if loss_fn is None:
            loss = model.compute_supervised_loss(logits, labels)
        else:
            loss = loss_fn(logits, labels)
        
        total_loss += loss.item() * labels.shape[0]
        
        # Compute accuracy
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.shape[0]
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'num_samples': total_samples
    }


class ContrastivePretrainingScheduler:
    """Scheduler for contrastive pre-training phases.
    
    Typically, we want to:
    1. Pre-train with contrastive learning (self-supervised) for N epochs
    2. Fine-tune with supervised learning
    3. (Optional) Fine-tune with combined learning
    """
    
    def __init__(self, num_pretrain_epochs: int = 50, num_finetune_epochs: int = 20):
        self.num_pretrain_epochs = num_pretrain_epochs
        self.num_finetune_epochs = num_finetune_epochs
        self.phase = 'pretrain'
        self.epoch = 0
    
    def step_epoch(self):
        """Advance to next epoch."""
        self.epoch += 1
        
        if self.epoch >= self.num_pretrain_epochs:
            self.phase = 'finetune'
    
    def get_current_phase(self) -> str:
        """Get current training phase."""
        return self.phase


# Example usage and recommended training workflow
"""
EXAMPLE: Complete training pipeline with SimCLR

from models.heavy_smart.model import build_model
from models.heavy_smart.simclr_training import (
    train_epoch_contrastive,
    train_epoch_supervised,
    train_epoch_combined,
    evaluate_supervised,
    ContrastivePretrainingScheduler
)

# Initialize model
model = build_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler_obj = ContrastivePretrainingScheduler(num_pretrain_epochs=50, num_finetune_epochs=20)

# ==== Phase 1: Contrastive Pre-training (self-supervised) ====
for epoch in range(scheduler_obj.num_pretrain_epochs):
    stats = train_epoch_contrastive(
        model, train_loader, optimizer, device,
        augmentation_level="moderate"
    )
    print(f"Epoch {epoch+1}: Contrastive Loss = {stats['contrastive_loss']:.4f}")

# ==== Phase 2: Supervised Fine-tuning ====
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
for epoch in range(scheduler_obj.num_finetune_epochs):
    stats = train_epoch_supervised(model, train_loader, optimizer, device)
    val_stats = evaluate_supervised(model, test_loader, device)
    print(f"Epoch {epoch+1}: Loss = {stats['loss']:.4f}, Acc = {val_stats['accuracy']:.4f}")

# ==== (Optional) Phase 3: Combined Training ====
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(10):
    stats = train_epoch_combined(
        model, train_loader, optimizer, device,
        supervised_weight=1.0,
        contrastive_weight=0.1,  # Lower weight in fine-tuning phase
        augmentation_level="light"
    )
    print(f"Epoch {epoch+1}: Total = {stats['total_loss']:.4f}, Sup = {stats['supervised_loss']:.4f}, Cont = {stats['contrastive_loss']:.4f}")
"""
