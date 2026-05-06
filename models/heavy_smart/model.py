import csv
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


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
    """Per-subaction module that emits a 4-element encoded representation."""

    def __init__(self, in_channels: int, out_classes: int, representation: str, class_values=None):
        super().__init__()
        self.representation = representation
        self.video_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(64, out_classes)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

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
        logits = self.final_fc(attended.squeeze(1))
        return logits


def build_model(cfg=None):
    subaction_counts = None
    if cfg is not None:
        try:
            # optional hook: cfg may provide explicit subaction class counts
            subaction_counts = cfg.get_subaction_class_counts()
        except Exception:
            subaction_counts = None
    return Heavy(subaction_counts=subaction_counts)