import argparse
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


# -----------------------------
# Tensor video preprocessing
# -----------------------------
def resize_short_side(video, short_side=256):
    # video: [T, C, H, W]
    t, c, h, w = video.shape

    if h < w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))

    return F.interpolate(
        video,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )


def center_crop(video, crop_size=224):
    _, _, h, w = video.shape
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    return video[:, :, top:top + crop_size, left:left + crop_size]


def random_resized_crop(video, crop_size=224, scale=(0.75, 1.0), ratio=(0.9, 1.1)):
    # video: [T, C, H, W]
    _, _, h, w = video.shape
    area = h * w

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(*ratio)

        crop_h = int(round((target_area / aspect) ** 0.5))
        crop_w = int(round((target_area * aspect) ** 0.5))

        if 0 < crop_h <= h and 0 < crop_w <= w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            crop = video[:, :, top:top + crop_h, left:left + crop_w]

            return F.interpolate(
                crop,
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            )

    return center_crop(resize_short_side(video, 256), crop_size)


def color_jitter_video(video, brightness=0.2, contrast=0.2, saturation=0.15):
    # video: [T, C, H, W], values in [0, 1]
    if brightness > 0:
        b = random.uniform(max(0, 1 - brightness), 1 + brightness)
        video = video * b

    if contrast > 0:
        c = random.uniform(max(0, 1 - contrast), 1 + contrast)
        mean = video.mean(dim=(2, 3), keepdim=True)
        video = (video - mean) * c + mean

    if saturation > 0:
        s = random.uniform(max(0, 1 - saturation), 1 + saturation)
        gray = video.mean(dim=1, keepdim=True)
        video = (video - gray) * s + gray

    return video.clamp(0.0, 1.0)


# -----------------------------
# Dataset
# -----------------------------
class Diving48TensorDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        tensor_dir,
        split,
        max_samples=None,
        hflip=False,
    ):
        self.annotation_path = Path(annotation_path)
        self.tensor_dir = Path(tensor_dir)
        self.split = split
        self.hflip = hflip

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
        print(f"Split: {self.split}")
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

        # Expected shape: [16, 3, H, W]
        x = torch.load(tensor_path, map_location="cpu")

        if x.ndim != 4 or x.shape[0] != 16 or x.shape[1] != 3:
            raise RuntimeError(f"Bad shape for {tensor_path}: got {x.shape}, expected [16, 3, H, W]")

        x = x.float()

        if x.max() > 2.0:
            x = x / 255.0

        if self.split == "train":
            x = color_jitter_video(x, brightness=0.2, contrast=0.2, saturation=0.15)

            if self.hflip and random.random() < 0.5:
                x = torch.flip(x, dims=[3])

        x = (x - self.mean) / self.std

        return x, label


# -----------------------------
# Model
# -----------------------------
class ResNet50TemporalConvVideoClassifier(nn.Module):
    def __init__(self, num_classes=48, pretrained=True, dropout=0.3):
        super().__init__()

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

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        frame_features = self.backbone(x)          # [B*T, 2048]
        frame_features = self.frame_projector(frame_features)  # [B*T, 512]

        frame_features = frame_features.view(b, t, 512)        # [B, T, 512]
        frame_features = frame_features.transpose(1, 2)        # [B, 512, T]

        temporal_features = self.temporal_conv(frame_features) # [B, 512, T]
        video_features = temporal_features.mean(dim=2)         # [B, 512]

        logits = self.classifier(video_features)
        return logits


# -----------------------------
# Freeze / unfreeze policy
# -----------------------------
def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_layer4(model):
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True


def unfreeze_layer3(model):
    for p in model.backbone.layer3.parameters():
        p.requires_grad = True


def set_trainable_by_epoch(model, epoch):
    freeze_backbone(model)

    if epoch >= 4:
        unfreeze_layer4(model)

    if epoch >= 13:
        unfreeze_layer3(model)


def build_optimizer(model, backbone_lr, head_lr, weight_decay):
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        })

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        })

    return torch.optim.AdamW(param_groups)


# -----------------------------
# Train / eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, label_smoothing):
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, label_smoothing, grad_clip, limit_batches=None):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

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


def save_checkpoint(model, optimizer, epoch, args, path, best_acc):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
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

    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--num-classes", type=int, default=48)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--hflip", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default="/home/jacktuck/DivingVision/outputs")
    parser.add_argument("--run-name", type=str, default="resnet50_16f_temporal_conv")

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
        split="train",
        max_samples=args.max_train_samples,
        hflip=args.hflip,
    )

    test_dataset = Diving48TensorDataset(
        test_json,
        tensor_dir,
        split="test",
        max_samples=args.max_test_samples,
        hflip=False,
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

    model = ResNet50TemporalConvVideoClassifier(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    best_acc = -1.0
    optimizer = None

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        set_trainable_by_epoch(model, epoch)

        optimizer = build_optimizer(
            model,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"epoch={epoch:03d} trainable_params={trainable_params:,}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            limit_batches=args.limit_train_batches,
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            label_smoothing=args.label_smoothing,
        )

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
        save_checkpoint(model, optimizer, epoch, args, last_ckpt, best_acc)

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            best_ckpt = output_dir / "best.pt"
            save_checkpoint(model, optimizer, epoch, args, best_ckpt, best_acc)
            print(f"Saved new best checkpoint: {best_ckpt}")
            print(f"Best checkpoint accuracy so far: {best_acc:.4f}")

    print("Done.")
    print(f"Best checkpoint accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()