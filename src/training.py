import os

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.visualize import BarGraphVisualizer


def _prepare_batch(images):
    """Prepare a batch for image-based models.

    If `images` is a list of variable-length clips or a tensor of clips
    (B, T, H, W, C) this helper selects the middle frame from each clip
    and returns a stacked tensor of shape (B, C, H, W).
    """
    # If already a tensor of images (B, C, H, W) or (B, H, W, C)
    if isinstance(images, torch.Tensor):
        if images.ndim == 4:
            # Could be (B, C, H, W) or (B, H, W, C). Detect channel dim.
            if images.shape[1] in (1, 3):
                return images.float()
            elif images.shape[-1] in (1, 3):
                # (B, H, W, C) -> (B, C, H, W)
                imgs = images.permute(0, 3, 1, 2).float()
                return imgs
        elif images.ndim == 5:
            # (B, T, H, W, C) or (B, T, C, H, W)
            if images.shape[-1] in (1, 3):
                # (B, T, H, W, C)
                t = images.shape[1] // 2
                mid = images[:, t]
                return mid.permute(0, 3, 1, 2).float()
            elif images.shape[2] in (1, 3):
                # (B, T, C, H, W)
                t = images.shape[1] // 2
                mid = images[:, t]
                return mid.float()

    # If images is a list (variable-length clips)
    if isinstance(images, (list, tuple)):
        processed = []
        for s in images:
            if not isinstance(s, torch.Tensor):
                # Convert numpy array to tensor efficiently
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s).float()
                else:
                    s = torch.from_numpy(np.asarray(s)).float()
            # clip could be (T, H, W, C) or (T, C, H, W) or (H, W, C)
            if s.ndim == 4:
                # (T, H, W, C)
                t = s.shape[0] // 2
                frame = s[t]
                frame = frame.permute(2, 0, 1).float()
            elif s.ndim == 3:
                # (H, W, C)
                frame = s.permute(2, 0, 1).float()
            elif s.ndim == 5:
                # (B, T, H, W, C) unlikely per-sample, take mid
                t = s.shape[1] // 2
                frame = s[:, t].permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unsupported sample shape: {s.shape}")
            processed.append(frame)
        return torch.stack(processed, dim=0)

    raise ValueError("Unsupported batch images format")


# Train model
def _evaluate_accuracy(model, data_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            imgs = _prepare_batch(images)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return (correct / total) if total > 0 else 0.0


# Train model
def train_model(
    model,
    train_loader,
    test_loader,
    device=None,
    num_epochs=30,
    visualize_history=True,
    history_save_path=None,
    learning_rate=0.01,
    gradient_accumulation_steps=1,
    use_mixed_precision=True,
):
    """Train model with optimizations: GPU acceleration, mixed precision, SGD optimizer, LR scheduling.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Testing data loader
        device: Device to use (auto-detect if None)
        num_epochs: Number of epochs to train
        visualize_history: Whether to plot training history
        history_save_path: Path to save training history plot
        learning_rate: Initial learning rate for SGD
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_mixed_precision: Whether to use FP16 mixed precision training
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Use SGD with momentum (faster and more memory-efficient than Adam)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler: cosine annealing for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Mixed precision training for faster GPU operations
    scaler = GradScaler() if use_mixed_precision else None
    
    criterion = nn.CrossEntropyLoss()
    visualizer = BarGraphVisualizer()

    epochs = []
    train_losses = []
    test_accuracies = []

    print(f"Training on device: {device}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            imgs = _prepare_batch(images)
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                scaler.scale(loss).backward()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * imgs.size(0) * gradient_accumulation_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        # Learning rate scheduling
        scheduler.step()
        
        denom = len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1
        epoch_loss = running_loss / denom
        epochs.append(epoch + 1)
        train_losses.append(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.2e}")

        if test_loader is not None:
            model.eval()
            test_accuracy = _evaluate_accuracy(model, test_loader, device)
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy: {test_accuracy*100:.2f}%")
            model.train()

    model.eval()
    if test_loader is not None and not test_accuracies:
        accuracy = _evaluate_accuracy(model, test_loader, device)
        print(f"Test accuracy: {accuracy*100:.2f}%")
    elif test_loader is None:
        accuracy = 0.0

    if visualize_history and epochs:
        save_path = history_save_path or os.path.join("results", "training_history.png")
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        visualizer.plot_training_history(
            epochs=epochs,
            train_losses=train_losses,
            test_accuracies=test_accuracies if test_accuracies else None,
            save_path=save_path,
        )

    return model