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


def _fake_quantize(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Simple per-tensor fake quantization with straight-through estimator.

    This maps values to discrete levels but preserves gradient flow via
    the detach trick (STE).
    """
    if bits >= 32:
        return tensor
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min < 1e-8:
        return tensor
    q_levels = float(2 ** bits - 1)
    tensor_normalized = (tensor - t_min) / (t_max - t_min)
    tensor_q = torch.round(tensor_normalized * q_levels) / q_levels
    tensor_dequant = tensor_q * (t_max - t_min) + t_min
    return tensor + (tensor_dequant - tensor).detach()


class SubactionHead(nn.Module):
    """Per-subaction module: attention + LSTM + classifier returning softmax logits."""

    def __init__(self, embed_dim: int, lstm_hidden: int, out_classes: int, quant_bits: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, out_classes)
        self.quant_bits = quant_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # attention (we let the module compute attention over time)
        attn_out, _ = self.attn(x, x, x)
        # LSTM temporal encoder
        # Note: we avoid assigning Parameter objects in-place. LSTM weights
        # are not modified directly here; we perform fake quantization on
        # final linear layers where it is straightforward to simulate (STE).
        lstm_out, _ = self.lstm(attn_out)
        last = lstm_out[:, -1, :]
        if self.training:
            wq = _fake_quantize(self.fc.weight, self.quant_bits)
            b = self.fc.bias
            logits = F.linear(last, wq, b)
        else:
            logits = self.fc(last)
        probs = F.softmax(logits, dim=-1)
        return probs


class Heavy(nn.Module):
    """Temporal clip classifier composed of per-frame encoder and four subaction heads.

    The model encodes frames with a small CNN, produces per-frame embeddings,
    runs four dedicated heads (takeoff, somersaults, twists, position) that
    each return a softmax over their local classes. Those softmax vectors are
    concatenated and passed to a final FC that predicts the final `class_id`.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, lstm_hidden: int = 128,
                 quant_bits: int = 8, subaction_counts: Tuple[int, int, int, int, int] = None):
        super().__init__()
        if subaction_counts is None:
            subaction_counts = _infer_subaction_class_counts("translations.csv")
        takeoff_c, somersaults_c, twists_c, position_c, final_c = subaction_counts

        # small per-frame CNN encoder -> embed_dim features
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # per-subaction heads
        self.head_takeoff = SubactionHead(embed_dim, lstm_hidden, takeoff_c, quant_bits)
        self.head_somersaults = SubactionHead(embed_dim, lstm_hidden, somersaults_c, quant_bits)
        self.head_twists = SubactionHead(embed_dim, lstm_hidden, twists_c, quant_bits)
        self.head_position = SubactionHead(embed_dim, lstm_hidden, position_c, quant_bits)

        concat_dim = takeoff_c + somersaults_c + twists_c + position_c
        self.final_fc = nn.Linear(concat_dim, final_c)
        self.quant_bits = quant_bits

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> returns (B, T, D)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.frame_encoder(x)
        x = x.mean(dim=(-1, -2))
        x = x.view(B, T, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")
        embeddings = self._encode_frames(x)

        # Each head returns softmax probabilities for its subaction
        p_takeoff = self.head_takeoff(embeddings)
        p_somersaults = self.head_somersaults(embeddings)
        p_twists = self.head_twists(embeddings)
        p_position = self.head_position(embeddings)

        # concatenate probabilities and predict final class
        concat = torch.cat([p_takeoff, p_somersaults, p_twists, p_position], dim=-1)

        if self.training:
            wq = _fake_quantize(self.final_fc.weight, self.quant_bits)
            logits = F.linear(concat, wq, self.final_fc.bias)
        else:
            logits = self.final_fc(concat)
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