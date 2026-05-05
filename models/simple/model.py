from torch import nn


class SimpleCNN(nn.Module):
    """Very small clip classifier for pipeline smoke tests.

    Expects clip tensors shaped (B, T, C, H, W). Each frame is encoded with a
    tiny CNN, reduced with a plain spatial mean, then averaged across time.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 48):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        batch_size, num_frames, channels, height, width = x.shape
        x = x.reshape(batch_size * num_frames, channels, height, width)
        x = self.encoder(x)
        x = x.mean(dim=(-1, -2))
        x = x.reshape(batch_size, num_frames, -1)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(cfg=None):
    """Build a Simple model instance for pipeline smoke tests."""
    num_classes = 48
    if cfg is not None:
        try:
            cats = cfg.get_labels().get("categories", [])
            if cats:
                num_classes = len(cats)
        except Exception:
            pass

    return SimpleCNN(num_classes=num_classes)
