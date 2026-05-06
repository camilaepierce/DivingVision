# SimCLR Contrastive Learning Integration

## Overview

The heavy_smart model now includes **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)** contrastive learning capabilities. This enables:

1. **Self-supervised Pre-training**: Learn representations from unlabeled video without explicit labels
2. **Multi-task Learning**: Combine supervised classification with contrastive learning
3. **Fine-tuning**: Transfer pre-trained representations to downstream tasks

## Architecture Components

### 1. Core Model Enhancements

The `Heavy` model class now includes:

- **Projection Head**: MLP layer that projects the intermediate representation to a lower-dimensional space where contrastive loss is computed
- **Contrastive Loss (NT-Xent)**: Normalized Temperature-scaled Cross Entropy loss for SimCLR
- **Dual Forward Paths**: 
  - Standard classification forward pass
  - Representation extraction with optional projection

### 2. Video Augmentation Functions

Available augmentation strategies for creating different views of the same video:

- `random_temporal_crop()`: Randomly sample temporal windows
- `random_spatial_crop()`: Random spatial cropping
- `random_temporal_shift()`: Temporal jittering
- `random_brightness_contrast()`: Brightness and contrast adjustment
- `augment_video_simclr()`: Combines augmentations with configurable intensity levels:
  - `"light"`: Minimal augmentation
  - `"moderate"`: Balanced augmentation (default)
  - `"strong"`: Heavy augmentation for robust learning

### 3. Loss Functions

#### ContrastiveLoss (NT-Xent)
```python
loss = ContrastiveLoss(temperature=0.07, batch_size=32)
```

- **Temperature parameter**: Controls softness of the contrastive loss (lower = sharper)
- **Batch size**: Determines number of negative pairs

## Usage Examples

### Example 1: Pure Contrastive Pre-training (Self-Supervised)

```python
from models.heavy_smart.model import build_model, augment_video_simclr
import torch
from torch import optim

# Initialize model
model = build_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    total_loss = 0.0
    for videos in train_loader:  # Just videos, no labels needed
        videos = videos.to(device)
        
        # Create two augmented views
        videos_aug1 = augment_video_simclr(videos, "moderate")
        videos_aug2 = augment_video_simclr(videos, "moderate")
        
        optimizer.zero_grad()
        
        # Compute contrastive loss
        loss = model.compute_contrastive_loss(videos_aug1, videos_aug2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Contrastive Loss = {total_loss/len(train_loader):.4f}")
```

### Example 2: Supervised Classification (Standard)

```python
# Using standard supervised training
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(videos)
        
        # Compute supervised loss
        loss = model.compute_supervised_loss(logits, labels)
        
        loss.backward()
        optimizer.step()
```

### Example 3: Combined Multi-task Learning

```python
# Train with both supervised and contrastive objectives
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(40):
    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Create augmented view
        videos_aug = augment_video_simclr(videos, "moderate")
        
        optimizer.zero_grad()
        
        # Compute combined loss
        loss_dict = model.compute_combined_loss(
            videos, videos_aug, labels,
            supervised_weight=1.0,      # Weight for classification loss
            contrastive_weight=0.5      # Weight for contrastive loss
        )
        
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()
        
        print(f"Total: {loss_dict['total']:.4f}, "
              f"Supervised: {loss_dict['supervised']:.4f}, "
              f"Contrastive: {loss_dict['contrastive']:.4f}")
```

### Example 4: Using the Training Utilities

```python
from models.heavy_smart.simclr_training import (
    train_epoch_contrastive,
    train_epoch_supervised,
    train_epoch_combined,
    evaluate_supervised,
    ContrastivePretrainingScheduler
)

# Phase 1: Self-supervised pre-training (50 epochs)
print("=== Phase 1: Contrastive Pre-training ===")
for epoch in range(50):
    stats = train_epoch_contrastive(
        model, train_loader, optimizer, device,
        augmentation_level="moderate"
    )
    print(f"Epoch {epoch+1}: Contrastive Loss = {stats['contrastive_loss']:.4f}")

# Phase 2: Supervised fine-tuning
print("\n=== Phase 2: Supervised Fine-tuning ===")
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR
for epoch in range(20):
    stats = train_epoch_supervised(model, train_loader, optimizer, device)
    val_stats = evaluate_supervised(model, test_loader, device)
    print(f"Epoch {epoch+1}: Loss = {stats['loss']:.4f}, Acc = {val_stats['accuracy']:.4f}")

# Phase 3: Combined training
print("\n=== Phase 3: Combined Training ===")
for epoch in range(10):
    stats = train_epoch_combined(
        model, train_loader, optimizer, device,
        supervised_weight=1.0,
        contrastive_weight=0.1,
        augmentation_level="light"
    )
    print(f"Epoch {epoch+1}: Total Loss = {stats['total_loss']:.4f}")
```

## API Reference

### Heavy Model Methods

#### `forward(x, return_representation=False)`
**Parameters:**
- `x` (Tensor): Video tensor of shape `(B, T, C, H, W)`
- `return_representation` (bool): If True, also return intermediate representation

**Returns:**
- `logits` (Tensor): Classification logits `(B, num_classes)`
- `representation` (Tensor, optional): Intermediate representation `(B, 4)`

#### `compute_supervised_loss(logits, labels)`
Computes standard cross-entropy loss for classification.

#### `compute_contrastive_loss(x_i, x_j)`
Computes SimCLR NT-Xent loss between two augmented views.

#### `compute_combined_loss(x, x_aug, labels, supervised_weight=1.0, contrastive_weight=0.5)`
Combines both losses for multi-task learning.

### Augmentation Functions

#### `augment_video_simclr(video, augmentation_level="moderate")`
**Parameters:**
- `video` (Tensor): Video tensor `(B, T, C, H, W)`
- `augmentation_level` (str): "light", "moderate", or "strong"

**Returns:**
- Augmented video tensor

## Recommended Training Strategies

### Strategy 1: Pre-train → Fine-tune
Best for limited labeled data or when you want to leverage unlabeled data:

1. **Pre-train on unlabeled data** (50-100 epochs with contrastive loss)
2. **Fine-tune on labeled data** (20-30 epochs with supervised loss)
3. **Optional: Polish** (5-10 epochs with combined loss at low learning rate)

### Strategy 2: Joint Training
Good when you have sufficient labeled data:

1. **Train with combined loss** (30-50 epochs)
   - Balanced supervised and contrastive objectives
   - Learns discriminative features while maintaining semantic meaning

### Strategy 3: Curriculum Learning
Progressive approach:

1. **Phase 1** (epochs 1-20): Heavy contrastive weighting (e.g., cont_weight=1.0, sup_weight=0.1)
2. **Phase 2** (epochs 21-40): Balanced weighting (cont_weight=0.5, sup_weight=1.0)
3. **Phase 3** (epochs 41-50): Supervised dominated (cont_weight=0.1, sup_weight=1.0)

## Hyperparameter Tuning

### Temperature (τ)
- **Default**: 0.07
- **Lower values** (0.01-0.05): Sharper distinctions, potentially unstable
- **Higher values** (0.1-0.5): Smoother gradients, potentially weaker learning

### Projection Dimension
- **Default**: 128
- **Smaller** (64): Faster, less expressive
- **Larger** (256+): More expressive, potentially slower

### Augmentation Intensity
- **Light**: Use when data is scarce or domain-specific
- **Moderate**: General-purpose (recommended)
- **Strong**: Use when you want very robust representations

### Loss Weights (Combined Training)
- **High supervised weight** (1.0): Prioritize classification accuracy
- **High contrastive weight** (0.5-1.0): Prioritize feature learning
- **Balanced** (1.0 vs 0.5): Good starting point

## Performance Tips

1. **Use contrastive pre-training** when you have abundant unlabeled video data
2. **Combine losses** when labeled data is limited (e.g., <1000 samples)
3. **Use strong augmentation** if you want more robust representations
4. **Lower learning rate** during fine-tuning phases
5. **Save checkpoints** during pre-training to enable resuming

## Key Differences from Standard Training

| Aspect | Standard | With SimCLR |
|--------|----------|-----------|
| Data requirement | Labeled data required | Can use unlabeled data |
| Pre-training | Not applicable | Can self-supervised pre-train |
| Loss function | Classification only | Classification + Contrastive |
| Data augmentation | Optional | Essential for contrastive loss |
| Computational cost | Lower | Higher (due to augmentation) |
| Typical accuracy improvement | Baseline | +2-5% with pre-training |

## Troubleshooting

### Issue: Contrastive loss not decreasing
- Check augmentation intensity - might be too aggressive
- Verify batch size is reasonable (>16 recommended)
- Try lower temperature value (e.g., 0.01)

### Issue: Combined training not improving
- Check loss weights balance
- Try reducing contrastive weight in later epochs
- Verify that representations are being learned (check intermediate features)

### Issue: Out of memory
- Reduce batch size
- Use `gradient_accumulation_steps` in training loop
- Reduce augmentation intensity (less computation)

## References

- SimCLR paper: https://arxiv.org/abs/2002.05709
- NT-Xent loss explanation: https://arxiv.org/abs/2004.11362
