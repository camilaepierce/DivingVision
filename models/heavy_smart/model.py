import csv
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _infer_subaction_class_counts(csv_path: str) -> Tuple[int, int, int, int, int]:
    """Read translations.csv and return (takeoff, somersaults, twists, position, final_classes).

    If the file is missing or can't be parsed, fallback to sensible defaults.
    """
    p = Path(csv_path)
    if not p.exists():
        return 4, 9, 6, 4, 48

    vals = {"takeoff": set(), "somersaults": set(), "twists": set(), "position": set(), "class_id": set()}
    try:
        with p.open("r") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                # normalize header keys and strip values
                row = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() }
                # collect only non-empty values
                for key in vals.keys():
                    val = row.get(key)
                    if val is None or val == "":
                        continue
                    vals[key].add(val)
        return len(vals["takeoff"]), len(vals["somersaults"]), len(vals["twists"]), len(vals["position"]), len(vals["class_id"])
    except Exception:
        return 4, 9, 6, 4, 48


def _infer_subaction_values(csv_path: str, column: str):
    p = Path(csv_path)
    if not p.exists():
        return []

    values = set()
    try:
        with p.open("r") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items()}
                value = row.get(column)
                if value is None or value == "":
                    continue
                values.add(float(value))
    except Exception:
        return []

    return sorted(values)


# ============================================================================
# Video Augmentation Functions for SimCLR
# ============================================================================

def random_temporal_crop(video: torch.Tensor, target_frames: int = 30) -> torch.Tensor:
    """Randomly crop video to target_frames along temporal dimension.
    
    Args:
        video: Tensor of shape (B, T, C, H, W)
        target_frames: Target number of frames
        
    Returns:
        Cropped video tensor (B, target_frames, C, H, W)
    """
    if video.shape[1] <= target_frames:
        return video
    
    max_start = video.shape[1] - target_frames
    start = torch.randint(0, max_start + 1, (1,)).item()
    return video[:, start:start + target_frames, :, :, :]


def random_spatial_crop(video: torch.Tensor, crop_size: int = 200) -> torch.Tensor:
    """Randomly crop frames spatially.
    
    Args:
        video: Tensor of shape (B, T, C, H, W)
        crop_size: Size of the cropped region
        
    Returns:
        Cropped video tensor (B, T, C, crop_size, crop_size)
    """
    B, T, C, H, W = video.shape
    if H <= crop_size and W <= crop_size:
        return video
    
    max_h = max(0, H - crop_size)
    max_w = max(0, W - crop_size)
    
    h_start = torch.randint(0, max_h + 1, (1,)).item() if max_h > 0 else 0
    w_start = torch.randint(0, max_w + 1, (1,)).item() if max_w > 0 else 0
    
    return video[:, :, :, h_start:h_start + crop_size, w_start:w_start + crop_size]


def random_temporal_shift(video: torch.Tensor, shift_range: int = 2) -> torch.Tensor:
    """Randomly shift frames in time (jittering).
    
    Args:
        video: Tensor of shape (B, T, C, H, W)
        shift_range: Maximum temporal shift (frames)
        
    Returns:
        Shifted video tensor
    """
    if shift_range == 0 or video.shape[1] <= 1:
        return video
    
    shift = torch.randint(-shift_range, shift_range + 1, (1,)).item()
    if shift > 0:
        # Shift forward
        return torch.cat([video[:, shift:, :, :, :], video[:, :shift, :, :, :]], dim=1)
    elif shift < 0:
        # Shift backward
        return torch.cat([video[:, shift:, :, :, :], video[:, :shift, :, :, :]], dim=1)
    return video


def random_brightness_contrast(video: torch.Tensor, brightness_range: float = 0.1, 
                               contrast_range: float = 0.1) -> torch.Tensor:
    """Randomly adjust brightness and contrast.
    
    Args:
        video: Tensor of shape (B, T, C, H, W), normalized or unnormalized
        brightness_range: Fraction to adjust brightness
        contrast_range: Fraction to adjust contrast
        
    Returns:
        Augmented video tensor
    """
    B, T, C, H, W = video.shape
    
    # Random brightness adjustment
    if brightness_range > 0:
        brightness = 1.0 + torch.empty(B, 1, 1, 1, 1).uniform_(-brightness_range, brightness_range)
        video = video * brightness.to(video.device)
    
    # Random contrast adjustment
    if contrast_range > 0:
        contrast = 1.0 + torch.empty(B, 1, 1, 1, 1).uniform_(-contrast_range, contrast_range)
        mean = video.mean(dim=(2, 3, 4), keepdim=True)
        video = (video - mean) * contrast.to(video.device) + mean
    
    return video


def augment_video_simclr(video: torch.Tensor, augmentation_level: str = "moderate") -> torch.Tensor:
    """Apply SimCLR-style augmentations to video.
    
    Args:
        video: Tensor of shape (B, T, C, H, W)
        augmentation_level: "light", "moderate", or "strong"
        
    Returns:
        Augmented video tensor
    """
    # Apply augmentations based on level
    if augmentation_level in ["moderate", "strong"]:
        video = random_temporal_crop(video, target_frames=30)
        video = random_spatial_crop(video, crop_size=200)
        video = random_brightness_contrast(video, brightness_range=0.1, contrast_range=0.1)
    
    if augmentation_level == "strong":
        video = random_temporal_shift(video, shift_range=3)
        video = random_brightness_contrast(video, brightness_range=0.2, contrast_range=0.2)
    
    return video.clamp(-1.0, 1.0) if video.max() <= 1.0 else video


# ============================================================================
# SimCLR Contrastive Loss
# ============================================================================

class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss for SimCLR.
    
    This loss encourages the model to produce similar representations for
    augmented views of the same sample, and different representations for
    different samples.
    """
    
    def __init__(self, temperature: float = 0.07, batch_size: int = 32):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z_i: Representations from first augmentation view (B, feat_dim)
            z_j: Representations from second augmentation view (B, feat_dim)
            
        Returns:
            Scalar loss value
        """
        B = z_i.shape[0]
        
        # Normalize representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate: [z_i, z_j] -> (2B, feat_dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity = torch.mm(z, z.T) / self.temperature
        
        # Create labels: positive pairs are (i, B+i) and (B+i, i)
        labels = torch.cat([
            torch.arange(B, 2*B),
            torch.arange(B)
        ]).to(z_i.device)
        
        # Mask out self-similarities
        mask = torch.eye(2*B, dtype=torch.bool).to(z_i.device)
        similarity = similarity.masked_fill(mask, -9e15)
        
        # Compute cross-entropy loss
        pos_mask = torch.zeros(2*B, 2*B, dtype=torch.bool).to(z_i.device)
        pos_mask[:B, B:] = torch.eye(B, dtype=torch.bool).to(z_i.device)
        pos_mask[B:, :B] = torch.eye(B, dtype=torch.bool).to(z_i.device)
        
        loss = 0.0
        for idx in range(2*B):
            # Positive pairs
            pos_sim = similarity[idx][pos_mask[idx]].sum()
            # All negatives (including the positive for numerical stability)
            all_sim = torch.logsumexp(similarity[idx], dim=0)
            loss += -pos_sim + all_sim
        
        return loss / (2*B)


# ============================================================================
# SimCLR Projection Head
# ============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR contrastive learning.
    
    Projects the representation from the model backbone to a lower-dimensional
    space where the contrastive loss is computed.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project representation to contrastive space.
        
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            Projected tensor (B, output_dim)
        """
        return self.net(x)


class SubactionHead(nn.Module):
    """Per-subaction module that emits a 4-element encoded representation."""

    def __init__(self, in_channels: int, out_classes: int, representation: str, class_values=None):
        super().__init__()
        self.representation = representation
        self.video_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(1, 2, 2)),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(32, out_classes)
        if class_values is not None:
            self.register_buffer("class_values", torch.as_tensor(class_values, dtype=torch.float32))
        else:
            self.class_values = None

    def _encode_one_hot(self, logits: torch.Tensor) -> torch.Tensor:
        indices = logits.argmax(dim=-1)
        return F.one_hot(indices, num_classes=logits.shape[-1]).to(dtype=logits.dtype)

    def _encode_top_values(self, logits: torch.Tensor) -> torch.Tensor:
        topk = min(4, logits.shape[-1])
        topk_indices = logits.topk(k=topk, dim=-1).indices
        values = self.class_values[topk_indices]
        if topk < 4:
            pad = values[..., -1:].expand(*values.shape[:-1], 4 - topk)
            values = torch.cat([values, pad], dim=-1)
        return torch.round(values * 2.0) / 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        features = self.video_encoder(x).flatten(1)
        logits = self.fc(features)
        if self.representation == "one_hot":
            return self._encode_one_hot(logits)
        if self.representation == "top_values":
            return self._encode_top_values(logits)
        raise ValueError(f"Unknown representation: {self.representation}")


class Heavy(nn.Module):
    """Temporal clip classifier with SimCLR contrastive learning support.

    Each subaction head consumes the full video tensor and emits a 4-element
    encoding. Takeoff and position are one-hot encodings; somersaults and twists
    emit their top four predicted values, rounded to the nearest half-twist.
    The four head outputs are stacked into a (B, 4, 4) tensor and combined with
    an attention head before the final class projection.
    
    Supports two training modes:
    - Supervised: Standard classification loss
    - Contrastive: SimCLR contrastive loss for self-supervised pre-training
    - Combined: Both supervised and contrastive losses (multi-task learning)
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, lstm_hidden: int = 128,
                 subaction_counts: Tuple[int, int, int, int, int] = None,
                 enable_simclr: bool = True, projection_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        if subaction_counts is None:
            subaction_counts = _infer_subaction_class_counts("translations.csv")
        takeoff_c, somersaults_c, twists_c, position_c, final_c = subaction_counts

        somersault_values = _infer_subaction_values("translations.csv", "somersaults")
        twist_values = _infer_subaction_values("translations.csv", "twists")
        if not somersault_values:
            somersault_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]
        if not twist_values:
            twist_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

        # per-subaction heads
        self.head_takeoff = SubactionHead(in_channels, takeoff_c, "one_hot")
        self.head_somersaults = SubactionHead(in_channels, somersaults_c, "top_values", somersault_values)
        self.head_twists = SubactionHead(in_channels, twists_c, "top_values", twist_values)
        self.head_position = SubactionHead(in_channels, position_c, "one_hot")

        self.final_query = nn.Parameter(torch.zeros(1, 1, 4))
        self.final_attn = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)
        self.final_fc = nn.Linear(4, final_c)
        
        # ====== SimCLR Components ======
        self.enable_simclr = enable_simclr
        self.temperature = temperature
        
        if self.enable_simclr:
            # Projection head for contrastive learning
            self.projection_head = ProjectionHead(
                input_dim=4,
                hidden_dim=64,
                output_dim=projection_dim
            )
            # Contrastive loss function
            self.contrastive_loss_fn = ContrastiveLoss(temperature=temperature)
        
        self.final_c = final_c
    
    def _get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the intermediate representation before final classification.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Attention-weighted representation (B, 4)
        """
        # Each head returns softmax probabilities for its subaction
        p_takeoff = self.head_takeoff(x)
        p_somersaults = self.head_somersaults(x)
        p_twists = self.head_twists(x)
        p_position = self.head_position(x)

        # stack subaction encodings into a 4x4 tensor
        stacked = torch.stack([p_takeoff, p_somersaults, p_twists, p_position], dim=1)

        # attention over the four subaction encodings
        query = self.final_query.expand(stacked.shape[0], -1, -1)
        attended, _ = self.final_attn(query, stacked, stacked)
        representation = attended.squeeze(1)
        
        return representation

    def forward(self, x: torch.Tensor, return_representation: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional representation extraction.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            return_representation: If True, also return intermediate representation
            
        Returns:
            logits: Classification logits (B, num_classes)
            representation: (optional) Intermediate representation (B, 4)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        representation = self._get_representation(x)
        logits = self.final_fc(representation)
        
        if return_representation:
            return logits, representation
        return logits
    
    def compute_supervised_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute standard supervised classification loss.
        
        Args:
            logits: Model predictions (B, num_classes)
            labels: Ground truth labels (B,)
            
        Returns:
            Classification loss (scalar)
        """
        return F.cross_entropy(logits, labels)
    
    def compute_contrastive_loss(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute SimCLR contrastive loss between two augmented views.
        
        Args:
            x_i: First augmented view (B, T, C, H, W)
            x_j: Second augmented view (B, T, C, H, W)
            
        Returns:
            Contrastive loss (scalar)
        """
        if not self.enable_simclr:
            raise RuntimeError("SimCLR is not enabled for this model")
        
        # Extract representations
        _, rep_i = self.forward(x_i, return_representation=True)
        _, rep_j = self.forward(x_j, return_representation=True)
        
        # Project to contrastive space
        z_i = self.projection_head(rep_i)
        z_j = self.projection_head(rep_j)
        
        # Compute contrastive loss
        return self.contrastive_loss_fn(z_i, z_j)
    
    def compute_combined_loss(self, x: torch.Tensor, x_aug: torch.Tensor, labels: torch.Tensor,
                             supervised_weight: float = 1.0, contrastive_weight: float = 0.5) -> Dict[str, torch.Tensor]:
        """Compute combined supervised and contrastive loss (multi-task learning).
        
        Args:
            x: Original input (B, T, C, H, W)
            x_aug: Augmented input (B, T, C, H, W)
            labels: Ground truth labels (B,)
            supervised_weight: Weight for supervised loss
            contrastive_weight: Weight for contrastive loss
            
        Returns:
            Dictionary with 'total', 'supervised', and 'contrastive' losses
        """
        if not self.enable_simclr:
            raise RuntimeError("SimCLR is not enabled for this model")
        
        # Supervised loss
        logits = self.forward(x)
        sup_loss = self.compute_supervised_loss(logits, labels)
        
        # Contrastive loss
        cont_loss = self.compute_contrastive_loss(x, x_aug)
        
        # Combined loss
        total_loss = supervised_weight * sup_loss + contrastive_weight * cont_loss
        
        return {
            'total': total_loss,
            'supervised': sup_loss,
            'contrastive': cont_loss
        }


def build_model(cfg=None):
    subaction_counts = None
    if cfg is not None:
        try:
            # optional hook: cfg may provide explicit subaction class counts
            subaction_counts = cfg.get_subaction_class_counts()
        except Exception:
            subaction_counts = None
    return Heavy(subaction_counts=subaction_counts, enable_simclr=True)