import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


class Diving48TensorDataset(Dataset):
    def __init__(self, annotation_path, tensor_dir, max_samples=None):
        self.annotation_path = Path(annotation_path)
        self.tensor_dir = Path(tensor_dir)

        with open(self.annotation_path, "r") as f:
            anns = json.load(f)

        kept = []
        missing = 0

        for ann in anns:
            vid_name = ann["vid_name"]
            label = int(ann["label"])
            tensor_path = self.tensor_dir / f"{vid_name}.pt"

            if tensor_path.exists():
                kept.append((tensor_path, label, vid_name))
            else:
                missing += 1

        if max_samples is not None:
            rng = np.random.default_rng(1234)
            indices = rng.choice(len(kept), size=min(max_samples, len(kept)), replace=False)
            kept = [kept[i] for i in indices]

        self.samples = kept

        print(f"Loaded annotations from: {self.annotation_path}")
        print(f"Found usable samples: {len(self.samples)}")
        print(f"Missing tensor files skipped: {missing}")

        if len(self.samples) == 0:
            raise RuntimeError("No usable samples found. Check vid_name <-> .pt filename matching.")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label, vid_name = self.samples[idx]

        # Expected shape: [16, 3, 224, 224]
        x = torch.load(tensor_path, map_location="cpu")

        if x.shape != (16, 3, 224, 224):
            raise RuntimeError(f"Bad shape for {tensor_path}: got {x.shape}, expected [16, 3, 224, 224]")

        x = x.float()

        if x.max() > 2.0:
            x = x / 255.0

        x = (x - self.mean) / self.std

        return x, label


class ResNet50TemporalTransformerVideoClassifier(nn.Module):
    def __init__(
        self,
        num_classes=48,
        pretrained=True,
        num_frames=16,
        d_model=512,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = resnet50(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.frame_proj = nn.Linear(feature_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        # Run ResNet over individual frames.
        x = x.reshape(b * t, c, h, w)
        frame_features = self.backbone(x)

        # Convert frame features into temporal tokens.
        frame_features = frame_features.reshape(b, t, -1)
        frame_tokens = self.frame_proj(frame_features)

        # Add CLS token and positional embedding.
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, frame_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : t + 1, :]

        # Temporal attention over the 16 frame tokens.
        tokens = self.temporal_encoder(tokens)

        video_features = self.norm(tokens[:, 0])
        logits = self.classifier(video_features)

        return logits


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    criterion = nn.CrossEntropyLoss()

    for videos, labels in tqdm(loader, desc="eval", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(videos)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.numel()
        total_loss += loss.item() * labels.numel()

    return {
        "loss": total_loss / total_seen,
        "acc": total_correct / total_seen,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, limit_batches=None):
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    start_time = time.time()

    for batch_idx, (videos, labels) in enumerate(tqdm(loader, desc=f"train epoch {epoch}", leave=False)):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(videos)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_seen += labels.numel()
        total_loss += loss.item() * labels.numel()

    elapsed = time.time() - start_time
    videos_per_sec = total_seen / elapsed if elapsed > 0 else 0.0

    return {
        "loss": total_loss / total_seen,
        "acc": total_correct / total_seen,
        "videos_per_sec": videos_per_sec,
        "elapsed_sec": elapsed,
        "videos_seen": total_seen,
    }


def save_checkpoint(model, optimizer, epoch, args, path):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/jacktuck/orcd/scratch/6s058_project/repos/mmaction2/data/diving48",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=48)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)

    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)

    parser.add_argument("--output-dir", type=str, default="/home/jacktuck/DivingVision/outputs")
    parser.add_argument("--run-name", type=str, default="resnet50_16f_temporal_transformer")

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    train_json = data_root / "annotations" / "Diving48_V2_train.json"
    test_json = data_root / "annotations" / "Diving48_V2_test.json"
    tensor_dir = data_root / "frame_tensors"

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Data root:", data_root)
    print("Train annotation:", train_json)
    print("Test annotation:", test_json)
    print("Tensor dir:", tensor_dir)
    print("Output dir:", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = Diving48TensorDataset(
        train_json,
        tensor_dir,
        max_samples=args.max_train_samples,
    )
    test_dataset = Diving48TensorDataset(
        test_json,
        tensor_dir,
        max_samples=args.max_test_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = ResNet50TemporalTransformerVideoClassifier(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        num_frames=16,
        d_model=args.transformer_dim,
        nhead=args.transformer_heads,
        num_layers=args.transformer_layers,
        dropout=args.transformer_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            limit_batches=args.limit_train_batches,
        )

        test_metrics = evaluate(model, test_loader, device)

        estimated_full_epoch_sec = 18000 / max(train_metrics["videos_per_sec"], 1e-9)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"videos_per_sec={train_metrics['videos_per_sec']:.2f} "
            f"estimated_18k_epoch_min={estimated_full_epoch_sec / 60:.1f}"
        )

        last_ckpt = output_dir / "last.pt"
        save_checkpoint(model, optimizer, epoch, args, last_ckpt)

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            best_ckpt = output_dir / "best.pt"
            save_checkpoint(model, optimizer, epoch, args, best_ckpt)
            print(f"Saved new best checkpoint: {best_ckpt}")

    print("Done.")
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()