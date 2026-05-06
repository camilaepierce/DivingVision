import csv
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


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


class SubactionHead(nn.Module):
    """Temporal subaction head with temporal conv stack and compact token projection."""

    def __init__(self, feature_dim: int, out_classes: int, token_dim: int, dropout: float = 0.3):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, out_classes)
        self.projector = nn.Linear(out_classes, token_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {frame_features.shape}")

        x = frame_features.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.mean(dim=2)

        logits = self.classifier(x)
        return self.projector(logits)


class Heavy(nn.Module):
    """Temporal clip classifier with frame encoder, subaction heads, and transformer fusion."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        subaction_counts: Tuple[int, int, int, int, int] = None,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        del in_channels
        del lstm_hidden

        if subaction_counts is None:
            subaction_counts = _infer_subaction_class_counts("translations.csv")
        takeoff_c, somersaults_c, twists_c, position_c, final_c = subaction_counts
        if embed_dim % 8 != 0:
            raise ValueError(f"embed_dim must be divisible by 8 for transformer nhead=8, got {embed_dim}")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.frame_projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_takeoff = SubactionHead(512, takeoff_c, token_dim=embed_dim, dropout=dropout)
        self.head_somersaults = SubactionHead(512, somersaults_c, token_dim=embed_dim, dropout=dropout)
        self.head_twists = SubactionHead(512, twists_c, token_dim=embed_dim, dropout=dropout)
        self.head_position = SubactionHead(512, position_c, token_dim=embed_dim, dropout=dropout)

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, final_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        frame_features = self.backbone(x)
        frame_features = self.frame_projector(frame_features)
        frame_features = frame_features.view(b, t, 512)

        p_takeoff = self.head_takeoff(frame_features)
        p_somersaults = self.head_somersaults(frame_features)
        p_twists = self.head_twists(frame_features)
        p_position = self.head_position(frame_features)

        stacked = torch.stack([p_takeoff, p_somersaults, p_twists, p_position], dim=1)
        fused = self.fusion_transformer(stacked)
        return self.fusion(fused.mean(dim=1))


def build_model(cfg=None):
    subaction_counts = None
    pretrained = True
    dropout = 0.3

    if cfg is not None:
        try:
            # optional hook: cfg may provide explicit subaction class counts
            subaction_counts = cfg.get_subaction_class_counts()
        except Exception:
            subaction_counts = None

        if isinstance(cfg, dict):
            pretrained = cfg.get("pretrained", pretrained)
            dropout = cfg.get("dropout", dropout)
        else:
            pretrained = getattr(cfg, "pretrained", pretrained)
            dropout = getattr(cfg, "dropout", dropout)

    return Heavy(subaction_counts=subaction_counts, pretrained=pretrained, dropout=dropout)