# %%
# %%
# %% Imports

import json
import time
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap



# %%
# %%
# %% Settings

DATA_ROOT = Path("/home/jacktuck/orcd/scratch/6s058_project/repos/mmaction2/data/diving48")
TRAIN_JSON = DATA_ROOT / "annotations" / "Diving48_V2_train.json"
TEST_JSON = DATA_ROOT / "annotations" / "Diving48_V2_test.json"
TENSOR_DIR = DATA_ROOT / "frame_tensors"
OUTPUT_ROOT = Path("/home/jacktuck/DivingVision/outputs/resnet50_semantic_sweep")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 48

EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 4

# Number of SimCLR pretraining epochs to run before supervised fine-tuning
PRETRAIN_EPOCHS = 10
PRETRAIN_EPOCHS = 5
NUM_FROZEN = 20

LR = 3e-5
CONTRASTIVE_TEMP = 0.1
SOFT_TARGET_TEMP = 0.25
SELF_BOOST = 1.0

UMAP_MAX_POINTS = 2500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    print('failed')
    pass

# %%
SWEEP_CONFIGS = [
    {
        "name": "soft0_contrastive_low",
        "lambda_soft_ce": 0.0,
        "lambda_contrastive": 0.01,
    },
    {
        "name": "soft0_contrastive_high",
        "lambda_soft_ce": 0.0,
        "lambda_contrastive": 0.05,
    },
    {
        "name": "soft_low_contrastive0",
        "lambda_soft_ce": 0.1,
        "lambda_contrastive": 0.0,
    },
    {
        "name": "soft_high_contrastive0",
        "lambda_soft_ce": 0.3,
        "lambda_contrastive": 0.0,
    },
    {
        "name": "soft_low_contrastive_low",
        "lambda_soft_ce": 0.1,
        "lambda_contrastive": 0.01,
    },
    {
        "name": "soft_high_contrastive_high",
        "lambda_soft_ce": 0.3,
        "lambda_contrastive": 0.05,
    },
]

# %%
# %%
# %% Diving48 class vocabulary

CLASS_VOCAB = [
  ["Back", "15som", "05Twis", "FREE"], 
  ["Back", "15som", "15Twis", "FREE"], 
  ["Back", "15som", "25Twis", "FREE"], 
  ["Back", "15som", "NoTwis", "PIKE"], 
  ["Back", "15som", "NoTwis", "TUCK"], 
  ["Back", "25som", "15Twis", "PIKE"], 
  ["Back", "25som", "25Twis", "PIKE"], 
  ["Back", "25som", "NoTwis", "PIKE"], 
  ["Back", "25som", "NoTwis", "TUCK"], 
  ["Back", "2som", "15Twis", "FREE"], 
  ["Back", "2som", "25Twis", "FREE"], 
  ["Back", "35som", "NoTwis", "PIKE"], 
  ["Back", "35som", "NoTwis", "TUCK"], 
  ["Back", "3som", "NoTwis", "PIKE"], 
  ["Back", "3som", "NoTwis", "TUCK"], 
  ["Back", "Dive", "NoTwis", "PIKE"], 
  ["Back", "Dive", "NoTwis", "TUCK"], 
  ["Forward", "15som", "1Twis", "FREE"], 
  ["Forward", "15som", "2Twis", "FREE"], 
  ["Forward", "15som", "NoTwis", "PIKE"], 
  ["Forward", "1som", "NoTwis", "PIKE"], 
  ["Forward", "25som", "1Twis", "PIKE"], 
  ["Forward", "25som", "2Twis", "PIKE"], 
  ["Forward", "25som", "3Twis", "PIKE"], 
  ["Forward", "25som", "NoTwis", "PIKE"], 
  ["Forward", "25som", "NoTwis", "TUCK"], 
  ["Forward", "35som", "NoTwis", "PIKE"], 
  ["Forward", "35som", "NoTwis", "TUCK"], 
  ["Forward", "45som", "NoTwis", "TUCK"], 
  ["Forward", "Dive", "NoTwis", "PIKE"], 
  ["Forward", "Dive", "NoTwis", "STR"], 
  ["Inward", "15som", "NoTwis", "PIKE"], 
  ["Inward", "15som", "NoTwis", "TUCK"], 
  ["Inward", "25som", "NoTwis", "PIKE"], 
  ["Inward", "25som", "NoTwis", "TUCK"], 
  ["Inward", "35som", "NoTwis", "TUCK"], 
  ["Inward", "Dive", "NoTwis", "PIKE"], 
  ["Reverse", "15som", "05Twis", "FREE"], 
  ["Reverse", "15som", "15Twis", "FREE"], 
  ["Reverse", "15som", "25Twis", "FREE"], 
  ["Reverse", "15som", "35Twis", "FREE"], 
  ["Reverse", "15som", "NoTwis", "PIKE"], 
  ["Reverse", "25som", "15Twis", "PIKE"], 
  ["Reverse", "25som", "NoTwis", "PIKE"], 
  ["Reverse", "25som", "NoTwis", "TUCK"], 
  ["Reverse", "35som", "NoTwis", "TUCK"], 
  ["Reverse", "Dive", "NoTwis", "PIKE"], 
  ["Reverse", "Dive", "NoTwis", "TUCK"]
]

# %%
# %%
# %% Semantic similarity helpers

def get_num_flips(s):
    if s == "Dive":
        return 0.5
    if s in ["15som", "25som", "35som", "45som"]:
        return float(s.replace("som", "")) / 10.0  
    else:
        return float(s.replace("som", ""))


def get_num_twists(s):
    if s == "NoTwis":
        return 0.0
    if s in ["05Twis", "15Twis", "25Twis", "35Twis"]:
        return float(s.replace("Twis", "")) / 10.0  
    else:
        return float(s.replace("Twis", ""))


def parse_class(v):
    return {
        "takeoff": v[0],
        "flips": get_num_flips(v[1]),
        "twists": get_num_twists(v[2]),
        "position": v[3],
    }

CLASS_ATTRS = [parse_class(v) for v in CLASS_VOCAB]


def numeric_action_similarity(a_val, b_val):
    '''
    Applies to num flips & num twists
    '''
    d = abs(a_val - b_val)

    # same rotation
    if d == 0:
        return 1.0
    
    # multiplier for large rotation difference
    if d >= 2.5:
        mult = 0.8
    elif d >= 1.5:
        mult = 0.9
    else:
        mult = 1.0

    # different direction on entry -> not similar
    if d%1 == 0.5:
        return 0.6*mult
    # same direction on entry -> similar
    elif d%1 == 0.0:
        return 0.9*mult
    


def position_similarity(pos_a, pos_b):
    if pos_a == pos_b:
        return 1.0

    pair = {pos_a, pos_b}

    # pike/straight look very similar
    if pair == {"PIKE", "STR"}:
        return 0.90
    # tuck/pike look somewhat similar
    elif pair == {"TUCK", "PIKE"}:
        return 0.7
    # tuck/straight not similar
    elif pair == {"TUCK", "STR"}:
        return 0.30
    # free is always a twister -> let twister logic handle 
    elif "FREE" in pair:
        return 1.0
    
    

def label_similarity(a, b):
    # different takeoffs look different
    if a["takeoff"] != b["takeoff"]:
        takeoff_sim = 0.5
    else:
        takeoff_sim = 1.0

    a_twisting = a["twists"] > 0
    b_twisting = b["twists"] > 0

    # no twists vs twisting dives look very different
    if a_twisting != b_twisting:
        twist_sim = 0.3
    else:
        twist_sim = numeric_action_similarity(a["twists"], b["twists"])
    

    pos_sim = position_similarity(a["position"], b["position"])
    flip_sim = numeric_action_similarity(a["flips"], b["flips"])


    return takeoff_sim * pos_sim * flip_sim * twist_sim


def build_similarity_matrix(class_vocab):
    attrs = [parse_class(v) for v in class_vocab]
    n = len(attrs)

    S = torch.zeros(n, n, dtype=torch.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = 1.0
            else:
                S[i, j] = label_similarity(attrs[i], attrs[j])

    return S


def build_soft_targets(sim_matrix, temperature=0.25, self_boost=1.0):
    S = sim_matrix.clone()

    for i in range(S.size(0)):
        S[i, i] = self_boost

    soft_targets = torch.softmax(S / temperature, dim=1)
    return soft_targets


SIM_MATRIX = build_similarity_matrix(CLASS_VOCAB).to(DEVICE)
SOFT_TARGETS = build_soft_targets(
    SIM_MATRIX,
    temperature=SOFT_TARGET_TEMP,
    self_boost=SELF_BOOST,
).to(DEVICE)


print("Example top similarity values:")
top_vals, top_idx = SIM_MATRIX[8].topk(8)
for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
    print(idx, CLASS_VOCAB[idx], round(val, 4))
    
print("\nExample soft target top values:")
top_vals, top_idx = SOFT_TARGETS[8].topk(8)
for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
    print(idx, CLASS_VOCAB[idx], round(val, 4))



# %%
# %%
# %% Dataset

class Diving48TensorDataset(Dataset):
    def __init__(self, annotation_path, tensor_dir, train=False):
        self.annotation_path = Path(annotation_path)
        self.tensor_dir = Path(tensor_dir)
        self.train = train

        with open(self.annotation_path, "r") as f:
            anns = json.load(f)

        self.samples = []
        missing = 0

        for ann in anns:
            vid_name = ann["vid_name"]
            label = int(ann["label"])
            tensor_path = self.tensor_dir / f"{vid_name}.pt"

            if tensor_path.exists():
                self.samples.append((tensor_path, label))
            else:
                missing += 1

        print(f"Loaded {len(self.samples)} samples from {self.annotation_path}")
        print(f"Missing tensor files skipped: {missing}")

        if len(self.samples) == 0:
            raise RuntimeError("No usable samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor_path, label = self.samples[idx]

        x = torch.load(tensor_path, map_location="cpu")
        x = x.float()

        if x.shape != (16, 3, 224, 224):
            raise RuntimeError(f"Bad shape for {tensor_path}: {x.shape}")

        if self.train and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])

        return x, label



# %%
# %%
# %% DataLoaders

train_dataset = Diving48TensorDataset(TRAIN_JSON, TENSOR_DIR, train=True)
test_dataset = Diving48TensorDataset(TEST_JSON, TENSOR_DIR, train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

# %%
# %%
# %% Model

class ResNet50MeanPoolClassifier(nn.Module):
    def __init__(self, num_classes=48, enable_simclr: bool = True, projection_dim: int = 128):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        for ix, param in enumerate(self.backbone.parameters()):
            if ix > NUM_FROZEN:
                param.requires_grad = False
        self.classifier = nn.Linear(feature_dim, num_classes)

        # SimCLR components
        self.enable_simclr = enable_simclr
        self.projection_dim = projection_dim
        if self.enable_simclr:
            # ProjectionHead is defined later in the file; it's fine to instantiate here
            self.projection_head = ProjectionHead(feature_dim, hidden_dim=512, output_dim=projection_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)

        feats = feats.view(b, t, -1)
        feats = feats.mean(dim=1)

        logits = self.classifier(feats)
        return logits, feats

    def compute_contrastive_loss(self, x_i: torch.Tensor, x_j: torch.Tensor, temperature: float = CONTRASTIVE_TEMP) -> torch.Tensor:
        """Compute NT-Xent loss between two augmented video batches using model's projection head.

        Args:
            x_i, x_j: (B, T, C, H, W)
        """
        if not hasattr(self, "projection_head"):
            # create a projection head on demand
            feature_dim = self.backbone.fc.in_features if hasattr(self.backbone, "fc") else self.backbone.fc.in_features
            self.projection_head = ProjectionHead(feature_dim, hidden_dim=512, output_dim=self.projection_dim)

        # forward to get features
        _l1, feats1 = self.forward(x_i)
        _l2, feats2 = self.forward(x_j)

        z1 = self.projection_head(feats1)
        z2 = self.projection_head(feats2)

        return nt_xent_loss(z1, z2, temperature=temperature)


# ============================================================================
# SimCLR utilities: augmentations, projection head, NT-Xent loss, pretrain
# ============================================================================


def random_temporal_crop(video: torch.Tensor, target_frames: int = 16) -> torch.Tensor:
    # video: (B, T, C, H, W)
    if video.shape[1] <= target_frames:
        return video
    max_start = video.shape[1] - target_frames
    start = torch.randint(0, max_start + 1, (1,)).item()
    return video[:, start:start + target_frames, :, :, :]


def random_spatial_crop(video: torch.Tensor, crop_size: int = 200) -> torch.Tensor:
    # video: (B, T, C, H, W)
    B, T, C, H, W = video.shape
    if H <= crop_size and W <= crop_size:
        return video
    max_h = max(0, H - crop_size)
    max_w = max(0, W - crop_size)
    h_start = torch.randint(0, max_h + 1, (1,)).item() if max_h > 0 else 0
    w_start = torch.randint(0, max_w + 1, (1,)).item() if max_w > 0 else 0
    return video[:, :, :, h_start:h_start + crop_size, w_start:w_start + crop_size]


def random_brightness_contrast(video: torch.Tensor, brightness_range: float = 0.1, contrast_range: float = 0.1) -> torch.Tensor:
    B, T, C, H, W = video.shape
    if brightness_range > 0:
        brightness = 1.0 + torch.empty(B, 1, 1, 1, 1).uniform_(-brightness_range, brightness_range)
        video = video * brightness.to(video.device)
    if contrast_range > 0:
        contrast = 1.0 + torch.empty(B, 1, 1, 1, 1).uniform_(-contrast_range, contrast_range)
        mean = video.mean(dim=(2, 3, 4), keepdim=True)
        video = (video - mean) * contrast.to(video.device) + mean
    return video


def augment_video_simclr(video: torch.Tensor, augmentation_level: str = "moderate") -> torch.Tensor:
    # Operate on (B, T, C, H, W)
    if augmentation_level in ["moderate", "strong"]:
        video = random_temporal_crop(video, target_frames=16)
        video = random_spatial_crop(video, crop_size=200)
        video = random_brightness_contrast(video, brightness_range=0.1, contrast_range=0.1)
    if augmentation_level == "strong":
        # small temporal jitter
        shift = torch.randint(-2, 3, (1,)).item()
        if shift > 0:
            video = torch.cat([video[:, shift:, :, :, :], video[:, :shift, :, :, :]], dim=1)
        elif shift < 0:
            video = torch.cat([video[:, shift:, :, :, :], video[:, :shift, :, :, :]], dim=1)
        video = random_brightness_contrast(video, brightness_range=0.2, contrast_range=0.2)
    return video.clamp(-1.0, 1.0) if video.max() <= 1.0 else video


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # z_i, z_j: (B, dim)
    B = z_i.shape[0]
    device = z_i.device

    z = torch.cat([F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)], dim=0)  # (2B, dim)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask out self similarities
    diag = torch.eye(2 * B, device=device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float('-inf'))

    positives = torch.cat([torch.arange(B, 2 * B, device=device), torch.arange(0, B, device=device)])

    # numerator: exp(sim[i, positives[i]])
    numerator = torch.exp(sim[torch.arange(2 * B, device=device), positives])

    # denominator: sum_j exp(sim[i, j]) over all j != i
    denom = torch.exp(sim).sum(dim=1)

    loss = -torch.log(numerator / (denom + 1e-12))
    return loss.mean()


def train_epoch_pretrain_semantic(model, train_loader, optimizer, device, augmentation_level="moderate"):
    """Pretrain one epoch using semantic_contrastive_loss on augmented videos.
    
    Features are trained to be closer for semantically similar classes.
    """
    model.train()
    total_loss = 0.0
    n = 0

    for videos, labels in tqdm(train_loader, desc="Semantic pretrain", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Augment videos
        videos_aug = augment_video_simclr(videos, augmentation_level)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass on augmented videos
        _logits, feats = model(videos_aug)

        # Compute semantic contrastive loss
        loss = semantic_contrastive_loss(
            feats,
            labels,
            SIM_MATRIX,
            temperature=CONTRASTIVE_TEMP,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"loss": total_loss / max(1, n), "num_batches": n}




# %%
# %%
# %% Losses

criterion = nn.CrossEntropyLoss()


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def semantic_contrastive_loss(feats, labels, sim_matrix, temperature=0.1):
    batch_size = feats.size(0)

    if batch_size <= 1:
        return feats.new_tensor(0.0)

    # Force this loss to float32 because masking with large negatives can overflow in fp16.
    z = F.normalize(feats.float(), dim=1)

    sim_logits = z @ z.T
    sim_logits = sim_logits / temperature

    eye = torch.eye(batch_size, device=feats.device, dtype=torch.bool)
    sim_logits = sim_logits.masked_fill(eye, -1e4)

    target_weights = sim_matrix[labels][:, labels].float().clone()
    target_weights = target_weights.masked_fill(eye, 0.0)

    row_sums = target_weights.sum(dim=1, keepdim=True)
    valid = row_sums.squeeze(1) > 0

    if valid.sum() == 0:
        return feats.new_tensor(0.0)

    target_weights = target_weights / row_sums.clamp_min(1e-8)

    log_probs = F.log_softmax(sim_logits, dim=1)
    loss_per_sample = -(target_weights * log_probs).sum(dim=1)

    return loss_per_sample[valid].mean()

# %%
# %% Metric helpers

def attr_match(true_labels, pred_labels, attr_name):
    correct = 0
    total = len(true_labels)

    for y, p in zip(true_labels, pred_labels):
        if CLASS_ATTRS[y][attr_name] == CLASS_ATTRS[p][attr_name]:
            correct += 1

    return correct / total


def twist_status_match(true_labels, pred_labels):
    correct = 0
    total = len(true_labels)

    for y, p in zip(true_labels, pred_labels):
        true_twist = CLASS_ATTRS[y]["twists"] > 0
        pred_twist = CLASS_ATTRS[p]["twists"] > 0

        if true_twist == pred_twist:
            correct += 1

    return correct / total


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history, output_dir):
    epochs = [r["epoch"] for r in history]

    plt.figure()
    plt.plot(epochs, [r["test_top1"] for r in history], label="Top-1")
    plt.plot(epochs, [r["test_top5"] for r in history], label="Top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Top-1 and Top-5 Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "top1_top5_over_epochs.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in history], label="Train total loss")
    plt.plot(epochs, [r["test_loss"] for r in history], label="Test CE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_over_epochs.png", dpi=200)
    plt.close()


def plot_error_severity(severities, output_dir):
    plt.figure()
    plt.hist(severities, bins=20)
    plt.xlabel("Error severity = 1 - semantic similarity")
    plt.ylabel("Count")
    plt.title("Error Severity Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "error_severity_histogram.png", dpi=200)
    plt.close()


def plot_umap(feats, labels, output_dir, name, color_values):

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    emb = reducer.fit_transform(feats)

    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=color_values, s=6)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=200)
    plt.close()


@torch.no_grad()
def collect_eval_outputs(model, loader, max_umap_points=None):
    model.eval()

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_seen = 0

    all_true = []
    all_pred = []
    all_top5 = []
    all_feats = []

    for videos, labels in tqdm(loader, desc="eval", leave=False):
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits, feats = model(videos)
        loss = criterion(logits, labels)

        top5 = logits.topk(5, dim=1).indices
        preds = top5[:, 0]

        total_loss += loss.item() * labels.size(0)
        total_top1 += (preds == labels).sum().item()
        total_top5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total_seen += labels.size(0)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_top5.extend(top5.cpu().tolist())

        if max_umap_points is None or len(all_feats) < max_umap_points:
            remaining = None if max_umap_points is None else max_umap_points - len(all_feats)
            take = feats if remaining is None else feats[:remaining]
            all_feats.extend(take.float().cpu().tolist())

    return {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
        "true": all_true,
        "pred": all_pred,
        "top5_preds": all_top5,
        "feats": all_feats,
    }


def compute_final_metrics(eval_out, output_dir):
    true = eval_out["true"]
    pred = eval_out["pred"]

    attr_rows = [
        {"metric": "takeoff_acc", "value": attr_match(true, pred, "takeoff")},
        {"metric": "flips_acc", "value": attr_match(true, pred, "flips")},
        {"metric": "twists_acc", "value": attr_match(true, pred, "twists")},
        {"metric": "twist_status_acc", "value": twist_status_match(true, pred)},
        {"metric": "position_acc", "value": attr_match(true, pred, "position")},
        {"metric": "full_class_top1", "value": eval_out["top1"]},
        {"metric": "full_class_top5", "value": eval_out["top5"]},
    ]

    save_csv(
        output_dir / "attribute_accuracy_table.csv",
        attr_rows,
        ["metric", "value"],
    )

    severities = []
    semantic_scores = []

    for y, p in zip(true, pred):
        if y != p:
            sim = SIM_MATRIX[y, p].item()
            semantic_scores.append(sim)
            severities.append(1.0 - sim)

    if len(severities) > 0:
        plot_error_severity(severities, output_dir)

    severity_summary = [
        {
            "mean_wrong_prediction_similarity": sum(semantic_scores) / max(1, len(semantic_scores)),
            "mean_error_severity": sum(severities) / max(1, len(severities)),
            "num_wrong": len(severities),
        }
    ]

    save_csv(
        output_dir / "error_severity_summary.csv",
        severity_summary,
        ["mean_wrong_prediction_similarity", "mean_error_severity", "num_wrong"],
    )

    if len(eval_out["feats"]) > 0:
        feats = torch.tensor(eval_out["feats"]).numpy()
        labels_for_umap = true[:len(feats)]

        takeoff_names = sorted(set(a["takeoff"] for a in CLASS_ATTRS))
        pos_names = sorted(set(a["position"] for a in CLASS_ATTRS))

        takeoff_to_id = {x: i for i, x in enumerate(takeoff_names)}
        pos_to_id = {x: i for i, x in enumerate(pos_names)}

        takeoff_colors = [takeoff_to_id[CLASS_ATTRS[y]["takeoff"]] for y in labels_for_umap]
        position_colors = [pos_to_id[CLASS_ATTRS[y]["position"]] for y in labels_for_umap]
        twist_colors = [1 if CLASS_ATTRS[y]["twists"] > 0 else 0 for y in labels_for_umap]
        flip_colors = [CLASS_ATTRS[y]["flips"] for y in labels_for_umap]

        plot_umap(feats, labels_for_umap, output_dir, "umap_by_takeoff", takeoff_colors)
        plot_umap(feats, labels_for_umap, output_dir, "umap_by_position", position_colors)
        plot_umap(feats, labels_for_umap, output_dir, "umap_by_twist_status", twist_colors)
        plot_umap(feats, labels_for_umap, output_dir, "umap_by_flips", flip_colors)

    return attr_rows, severity_summary




# %%
# %%
# %% Train/evaluate one sweep run

def train_one_epoch(model, loader, optimizer, scaler):
    """Finetune supervised epoch using semantic_contrastive_loss on frozen backbone features.

    The backbone is frozen; only the classifier is trained.
    Backbone features are trained to be closer for semantically similar classes.
    """
    model.train()

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_seen = 0

    start = time.time()

    for videos, labels in tqdm(loader, desc="train", leave=False):
        videos = videos.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits, feats = model(videos)

            # Use semantic contrastive loss on features
            loss = semantic_contrastive_loss(
                feats,
                labels,
                SIM_MATRIX,
                temperature=CONTRASTIVE_TEMP,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy from logits for reporting
        top5 = logits.topk(5, dim=1).indices
        preds = top5[:, 0]

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_top1 += (preds == labels).sum().item()
        total_top5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total_seen += bs

    elapsed = time.time() - start

    return {
        "train_loss": total_loss / total_seen,
        "train_top1": total_top1 / total_seen,
        "train_top5": total_top5 / total_seen,
        "videos_per_sec": total_seen / elapsed,
    }


def run_sweep(config):
    run_name = config["name"]
    lambda_soft_ce = config["lambda_soft_ce"]
    lambda_contrastive = config["lambda_contrastive"]

    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Run: {run_name}")
    print(f"lambda_soft_ce={lambda_soft_ce}")
    print(f"lambda_contrastive={lambda_contrastive}")
    print("=" * 80)

    # If config requests pretraining, enable simclr components on the model
    pretrain_epochs = PRETRAIN_EPOCHS #config.get("pretrain_epochs", 0)
    enable_simclr = pretrain_epochs > 0

    model = ResNet50MeanPoolClassifier(num_classes=NUM_CLASSES, enable_simclr=enable_simclr).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_top1 = -1.0
    best_top5 = -1.0
    best_epoch = -1
    history = []

    # ----- Optional: Contrastive pretraining phase -----
    if pretrain_epochs > 0:
        print(f"Starting SimCLR pretraining for {pretrain_epochs} epochs...")
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        for p_ep in range(1, pretrain_epochs + 1):
            stats = train_epoch_simclr(model, train_loader, pretrain_optimizer, DEVICE)
            print(f"Pretrain epoch={p_ep}/{pretrain_epochs} contrastive_loss={stats['contrastive_loss']:.4f}")
            # save periodic pretrain checkpoint
            torch.save(
                {
                    "epoch": p_ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": pretrain_optimizer.state_dict(),
                    "mode": "pretrain",
                    "config": config,
                },
                output_dir / f"pretrain_epoch_{p_ep:03d}.pt",
            )
        print("Pretraining complete.\n")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
        )

        eval_out = collect_eval_outputs(model, test_loader, max_umap_points=0)

        row = {
            "run_name": run_name,
            "epoch": epoch,
            **train_metrics,
            "test_loss": eval_out["loss"],
            "test_top1": eval_out["top1"],
            "test_top5": eval_out["top5"],
        }

        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_top1={row['train_top1']:.4f} "
            f"train_top5={row['train_top5']:.4f} "
            f"test_top1={row['test_top1']:.4f} "
            f"test_top5={row['test_top5']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_top1": best_top1,
                "best_top5": best_top5,
                "config": config,
                "class_vocab": CLASS_VOCAB,
                "sim_matrix": SIM_MATRIX.detach().cpu(),
                "soft_targets": SOFT_TARGETS.detach().cpu(),
            },
            output_dir / "last.pt",
        )

        if eval_out["top1"] > best_top1:
            best_top1 = eval_out["top1"]
            best_top5 = eval_out["top5"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_top1": best_top1,
                    "best_top5": best_top5,
                    "config": config,
                    "class_vocab": CLASS_VOCAB,
                    "sim_matrix": SIM_MATRIX.detach().cpu(),
                    "soft_targets": SOFT_TARGETS.detach().cpu(),
                },
                output_dir / "best.pt",
            )

            print(f"Saved best checkpoint: top1={best_top1:.4f}, top5={best_top5:.4f}")

    save_csv(
        output_dir / "history.csv",
        history,
        list(history[0].keys()),
    )

    plot_history(history, output_dir)

    # Reload best checkpoint for final detailed metrics.
    ckpt = torch.load(output_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    final_eval = collect_eval_outputs(
        model,
        test_loader,
        max_umap_points=UMAP_MAX_POINTS,
    )

    attr_rows, severity_summary = compute_final_metrics(final_eval, output_dir)

    summary = {
        "run_name": run_name,
        "lambda_soft_ce": lambda_soft_ce,
        "lambda_contrastive": lambda_contrastive,
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_top5_at_best_top1": best_top5,
        "final_top1": final_eval["top1"],
        "final_top5": final_eval["top5"],
        "mean_wrong_prediction_similarity": severity_summary[0]["mean_wrong_prediction_similarity"],
        "mean_error_severity": severity_summary[0]["mean_error_severity"],
    }

    return summary




# %%
# %%
# %% Run all sweeps

def run_pretrain_and_finetune(pretrain_epochs: int, finetune_epochs: int):
    output_dir = OUTPUT_ROOT / "simclr_single_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ResNet50MeanPoolClassifier(num_classes=NUM_CLASSES, enable_simclr=True).to(DEVICE)

    # ----- Pretraining with semantic contrastive loss -----
    if pretrain_epochs > 0:
        print(f"Starting semantic pretraining for {pretrain_epochs} epochs...")
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        for p_ep in range(1, pretrain_epochs + 1):
            stats = train_epoch_pretrain_semantic(model, train_loader, pretrain_optimizer, DEVICE)
            print(f"Pretrain epoch={p_ep}/{pretrain_epochs} loss={stats['loss']:.4f}")
            torch.save(
                {
                    "epoch": p_ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": pretrain_optimizer.state_dict(),
                    "mode": "pretrain",
                },
                output_dir / f"pretrain_epoch_{p_ep:03d}.pt",
            )
        # save final pretrained weights
        torch.save(model.state_dict(), output_dir / "pretrained_model.pt")
        print("Pretraining complete.\n")

    # ----- Freeze backbone after pretraining -----
    print("Freezing backbone weights...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen. Only classifier will be trained during fine-tuning.\n")

    # ----- Supervised fine-tuning (only classifier head) -----
    # Only optimize classifier since backbone is frozen
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_top1 = -1.0
    best_top5 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, finetune_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler)

        eval_out = collect_eval_outputs(model, test_loader, max_umap_points=0)

        row = {
            "epoch": epoch,
            **train_metrics,
            "test_loss": eval_out["loss"],
            "test_top1": eval_out["top1"],
            "test_top5": eval_out["top5"],
        }

        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_top1={row['train_top1']:.4f} "
            f"train_top5={row['train_top5']:.4f} "
            f"test_top1={row['test_top1']:.4f} "
            f"test_top5={row['test_top5']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            output_dir / "last.pt",
        )

        if eval_out["top1"] > best_top1:
            best_top1 = eval_out["top1"]
            best_top5 = eval_out["top5"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_top1": best_top1,
                    "best_top5": best_top5,
                },
                output_dir / "best.pt",
            )
            print(f"Saved best checkpoint: top1={best_top1:.4f}, top5={best_top5:.4f}")

    if len(history) > 0:
        save_csv(output_dir / "history.csv", history, list(history[0].keys()))
        plot_history(history, output_dir)

    # final evaluation and metrics
    if (output_dir / "best.pt").exists():
        ckpt = torch.load(output_dir / "best.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    final_eval = collect_eval_outputs(model, test_loader, max_umap_points=UMAP_MAX_POINTS)
    compute_final_metrics(final_eval, output_dir)

    print(f"Run complete. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    run_pretrain_and_finetune(PRETRAIN_EPOCHS, EPOCHS)

