import os

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.visualize import BarGraphVisualizer


def _prepare_batch(images):
    """Prepare a batch for image-based models.

    Returns a tensor shaped either:
    - (B, C, H, W) for single images
    - (B, T, C, H, W) for clips
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
                # (B, T, H, W, C) -> (B, T, C, H, W)
                return images.permute(0, 1, 4, 2, 3).float()
            elif images.shape[2] in (1, 3):
                return images.float()

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
                if s.shape[-1] in (1, 3):
                    # (T, H, W, C) -> (T, C, H, W)
                    frame = s.permute(0, 3, 1, 2).float()
                elif s.shape[1] in (1, 3):
                    # (T, C, H, W)
                    frame = s.float()
                else:
                    raise ValueError(f"Unsupported clip shape: {s.shape}")
            elif s.ndim == 3:
                # (H, W, C)
                frame = s.permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unsupported sample shape: {s.shape}")
            processed.append(frame)
        first = processed[0]
        if first.ndim == 3:
            return torch.stack(processed, dim=0)
        return torch.stack(processed, dim=0)

    raise ValueError("Unsupported batch images format")


# Train model
def _evaluate_accuracy(model, data_loader, device, max_batches=None, use_mixed_precision=False):
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            imgs = _prepare_batch(images)
            imgs = imgs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if use_mixed_precision:
                with autocast():
                    outputs = model(imgs)
            else:
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
    eval_every=1,
    max_eval_batches=None,
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
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Move model to device
    model = model.to(device)
    non_blocking = device.type == "cuda"
    use_mixed_precision = bool(use_mixed_precision and device.type == "cuda")
    
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
    print(f"Eval frequency (epochs): {eval_every}")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            imgs = _prepare_batch(images)
            imgs = imgs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            
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
                optimizer.zero_grad(set_to_none=True)

        # Flush gradients for the final partial accumulation window.
        if len(train_loader) % gradient_accumulation_steps != 0:
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Learning rate scheduling
        scheduler.step()
        
        denom = len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1
        epoch_loss = running_loss / denom
        epochs.append(epoch + 1)
        train_losses.append(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {current_lr:.2e}")

        should_eval = (
            test_loader is not None
            and ((epoch + 1) % max(1, int(eval_every)) == 0 or (epoch + 1) == num_epochs)
        )
        if should_eval:
            model.eval()
            test_accuracy = _evaluate_accuracy(
                model,
                test_loader,
                device,
                max_batches=max_eval_batches,
                use_mixed_precision=use_mixed_precision,
            )
            test_accuracies.append(test_accuracy)
            print(f"Test accuracy: {test_accuracy*100:.2f}%")
            model.train()

    model.eval()
    if test_loader is not None and not test_accuracies:
        accuracy = _evaluate_accuracy(
            model,
            test_loader,
            device,
            max_batches=max_eval_batches,
            use_mixed_precision=use_mixed_precision,
        )
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