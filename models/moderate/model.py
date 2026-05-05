from torch import nn


class FirstIteration(nn.Module):
  """Temporal clip classifier with a per-frame CNN encoder and LSTM head.

  This model expects clip tensors of shape (B, T, C, H, W) only.
  It rejects single-frame inputs and avoids adaptive pooling by reducing
  each frame's spatial feature map with a plain mean over H/W.
  """

  def __init__(self, in_channels: int = 3, num_classes: int = 48):
    super().__init__()

    self.frame_encoder = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
    )

    self.temporal_encoder = nn.LSTM(
      input_size=128,
      hidden_size=128,
      num_layers=1,
      batch_first=True,
    )

    self.classifier = nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(128, 64),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.4),
      nn.Linear(64, num_classes),
    )

  def forward(self, x):
    if x.ndim != 5:
      raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

    batch_size, num_frames, channels, height, width = x.shape
    x = x.reshape(batch_size * num_frames, channels, height, width)
    x = self.frame_encoder(x)
    x = x.mean(dim=(-1, -2))
    x = x.reshape(batch_size, num_frames, -1)

    temporal_output, _ = self.temporal_encoder(x)
    logits = self.classifier(temporal_output[:, -1, :])
    return logits


def build_model(cfg=None):
  num_classes = 48
  if cfg is not None:
    try:
      categories = cfg.get_labels().get("categories", [])
      if categories:
        num_classes = len(categories)
    except Exception:
      pass
  return FirstIteration(num_classes=num_classes)