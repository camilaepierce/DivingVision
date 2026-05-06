import csv
from pathlib import Path
from typing import Tuple

import torch
from torch import nn


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
    """Subaction head that maps shared clip features to a compact 4D embedding."""

    def __init__(self, feature_dim: int, out_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, out_classes)
        self.projector = nn.Linear(out_classes, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(features)
        return self.projector(logits)


class Heavy(nn.Module):
    """Temporal clip classifier composed of per-frame encoder and four subaction heads.

    Each subaction head consumes the full video tensor and emits a 4-element
    encoding. Takeoff and position are one-hot encodings; somersaults and twists
    emit their top four predicted values, rounded to the nearest half-twist.
    The four head outputs are stacked into a (B, 4, 4) tensor and combined with
    an attention head before the final class projection.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, lstm_hidden: int = 128,
                 subaction_counts: Tuple[int, int, int, int, int] = None):
        super().__init__()
        if subaction_counts is None:
            subaction_counts = _infer_subaction_class_counts("translations.csv")
        takeoff_c, somersaults_c, twists_c, position_c, final_c = subaction_counts

        # Shared clip encoder: run expensive 3D convolutions once per sample.
        self.video_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Lightweight per-subaction heads over shared features.
        self.head_takeoff = SubactionHead(32, takeoff_c)
        self.head_somersaults = SubactionHead(32, somersaults_c)
        self.head_twists = SubactionHead(32, twists_c)
        self.head_position = SubactionHead(32, position_c)

        self.fusion = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, final_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        clip_features = self.video_encoder(x).flatten(1)

        p_takeoff = self.head_takeoff(clip_features)
        p_somersaults = self.head_somersaults(clip_features)
        p_twists = self.head_twists(clip_features)
        p_position = self.head_position(clip_features)

        stacked = torch.stack([p_takeoff, p_somersaults, p_twists, p_position], dim=1)
        return self.fusion(stacked.flatten(1))


def build_model(cfg=None):
    subaction_counts = None
    if cfg is not None:
        try:
            # optional hook: cfg may provide explicit subaction class counts
            subaction_counts = cfg.get_subaction_class_counts()
        except Exception:
            subaction_counts = None
    return Heavy(subaction_counts=subaction_counts)