import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


class Diving48TensorDataset(Dataset):
    def __init__(self, annotation_path, tensor_dir, train=False, max_samples=None):
        self.annotation_path = Path(annotation_path)
        self.tensor_dir = Path(tensor_dir)
        self.train = train

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

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                224,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label, vid_name = self.samples[idx]

        # Expected saved tensor: [16, 3, H, W]
        x = torch.load(tensor_path, map_location="cpu")

        if x.ndim != 4 or x.shape[0] != 16 or x.shape[1] != 3:
            raise RuntimeError(
                f"Bad shape for {tensor_path}: got {x.shape}, expected [16, 3, H, W]"
            )

        x = x.float()

        # Handles tensors saved as uint8-like [0,255].
        if x.max() > 2.0:
            x = x / 255.0

        frames = []

        for frame in x:
            # frame: [3, H, W]
            if self.train:
                frame = self.train_transform(frame)
            else:
                frame = self.test_transform(frame)

            frames.append(frame)

        x = torch.stack(frames, dim=0)  # [16, 3, 224, 224]

        return x, label


class ResNet50MeanPoolDropoutClassifier(nn.Module):
    def __init__(self, num_classes=48, pretrained=True, dropout=0.4):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        frame_features = self.backbone(x)  # [B*T, 2048]

        frame_features = frame_features.view(b, t, -1)
        video_features = frame_features.mean(dim=1)  # [B, 2048]

        video_features = self.dropout(video_features)
        logits = self.classifier(video_features)

        return logits


def set_backbone_trainable(model, train_layer3=False, train_layer4=False):
    for param in model.backbone.parameters():
        param.requires_grad = False

    if train_layer4:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

    if train_layer3:
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


def print_trainable_params(model):
    total = 0
    trainable = 0

    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

    print(f"Trainable parameters: {trainable:,} / {total:,}")


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        })

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        })

    return torch.optim.AdamW(param_groups)


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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    label_smoothing,
    grad_clip,
    limit_batches=None,
):
    model.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    start_time = time.time()

    for batch_idx, (videos, labels) in enumerate(
        tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    ):
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
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )

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
        "args": vars(args),
        "best_acc": best_acc,
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
    parser.add_argument("--dropout", type=float, default=0.4)

    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-layer3-epoch", type=int, default=12)

    parser.add_argument("--num-classes", type=int, default=48)
    parser.add_argument("--no-pretrained", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default="/home/jacktuck/DivingVision/outputs")
    parser.add_argument("--run-name", type=str, default="resnet50_16f_meanpool_aug_dropout_disc_lr")

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
        train=True,
        max_samples=args.max_train_samples,
    )

    test_dataset = Diving48TensorDataset(
        test_json,
        tensor_dir,
        train=False,
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

    model = ResNet50MeanPoolDropoutClassifier(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = -1.0
    optimizer = None
    current_stage = None

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_epochs:
            stage = "head_only"
            train_layer4 = False
            train_layer3 = False
        elif epoch < args.unfreeze_layer3_epoch:
            stage = "layer4_plus_head"
            train_layer4 = True
            train_layer3 = False
        else:
            stage = "layer3_layer4_plus_head"
            train_layer4 = True
            train_layer3 = True

        if stage != current_stage:
            print(f"Switching training stage at epoch {epoch}: {stage}")

            set_backbone_trainable(
                model,
                train_layer3=train_layer3,
                train_layer4=train_layer4,
            )

            print_trainable_params(model)

            optimizer = make_optimizer(
                model,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
            )

            current_stage = stage

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
            f"stage={stage} "
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