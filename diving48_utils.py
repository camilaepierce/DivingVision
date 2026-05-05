import json
import os
import hashlib
import pickle
import tarfile
import tempfile
from io import BytesIO
import importlib.util
import inspect

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None

try:
    import cv2
except Exception:
    cv2 = None


class DivingConfig:
    def __init__(self, filename="config.json"):
        with open(filename) as f:
            self.config = json.load(f)

    def getAll(self):
        return self.config

    def getInfo(self):
        return self.config.get("standard", {})

    def trainingConfig(self):
        return self.config.get("training", {})

    def getDataConfig(self):
        return self.config.get("dataset", {})

    def get_training(self):
        return self.trainingConfig()

    def get_epochs(self):
        return int(self.get_training().get("epochs", 30))

    def get_batch_size(self):
        return int(self.get_training().get("batch_size", 8))

    def get_model_name(self):
        return self.get_training().get("model", None)

    def get_save_settings(self):
        train = self.get_training()
        return {"save_dir": train.get("save_dir", "results"), "save_model": train.get("save_model", False)}

    def get_labels(self):
        labels_cfg = self.config.get("labels", {})
        return {"categories": labels_cfg.get("categories", []), "label_map": labels_cfg.get("label_map", {})}


class DivingDataset(Dataset):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, data_config, split="train", transform=None, to_tensor=True,
                 frame_size=224, num_frames=16, cache_dir=None, normalize=True):
        self.data_config = data_config
        self.transform = transform
        self.to_tensor = to_tensor
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.normalize = normalize

        self.rgb_source = data_config.get("rgb_data")
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(self.rgb_source or "."), ".frame_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self._is_tar = False
        if self.rgb_source and os.path.isfile(self.rgb_source) and tarfile.is_tarfile(self.rgb_source):
            self._is_tar = True
            self._tar = tarfile.open(self.rgb_source, "r:*")
        else:
            self._tar = None

        split_path = data_config.get("train_split") if split == "train" else data_config.get("test_split")
        if not split_path:
            raise ValueError("Split path not provided in data_config")

        with open(split_path, "r") as f:
            self.entries = json.load(f)

    def _get_cache_path(self, vid_name: str, start: int, end: int) -> str:
        key = f"{vid_name}_{start}_{end}_{self.frame_size}_{self.num_frames}"
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")

    def _uniform_sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim < 4:
            raise ValueError(f"Expected clip tensor with time dimension, got shape {frames.shape}")
        total_frames = frames.shape[0]
        if total_frames == 0:
            raise ValueError("Cannot sample frames from an empty clip")
        if total_frames == self.num_frames:
            return frames
        if total_frames == 1:
            indices = np.zeros(self.num_frames, dtype=np.int64)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames)
            indices = np.round(indices).astype(np.int64)
        indices = np.clip(indices, 0, total_frames - 1)
        return frames[torch.from_numpy(indices)]

    def _load_from_cache(self, cache_path: str):
        try:
            with open(cache_path, "rb") as f:
                frames = pickle.load(f)
            return torch.from_numpy(frames).float()
        except Exception:
            return None

    def _save_to_cache(self, cache_path: str, frames: np.ndarray):
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(frames, f)
        except Exception:
            pass

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return frames
        mean = torch.tensor(self.IMAGENET_MEAN, device=frames.device, dtype=frames.dtype)
        std = torch.tensor(self.IMAGENET_STD, device=frames.device, dtype=frames.dtype)
        if frames.ndim == 4:
            if frames.shape[-1] == 3:
                mean = mean.reshape(1, 1, 1, 3)
                std = std.reshape(1, 1, 1, 3)
            else:
                mean = mean.reshape(1, 3, 1, 1)
                std = std.reshape(1, 3, 1, 1)
        elif frames.ndim == 3:
            if frames.shape[-1] == 3:
                mean = mean.reshape(1, 1, 3)
                std = std.reshape(1, 1, 3)
            else:
                mean = mean.reshape(3, 1, 1)
                std = std.reshape(3, 1, 1)
        return (frames - mean) / std

    def __len__(self):
        return len(self.entries)

    def _open_video_bytes(self, vid_name: str):
        filename = f"rgb/{vid_name}.mp4"
        if self._is_tar:
            try:
                member = self._tar.getmember(filename)
            except KeyError:
                member = self._tar.getmember(f"{vid_name}.mp4")
            fobj = self._tar.extractfile(member)
            return BytesIO(fobj.read())
        else:
            base = self.rgb_source
            candidate = os.path.join(base, "rgb", vid_name + ".mp4")
            if not os.path.exists(candidate):
                candidate = os.path.join(base, vid_name + ".mp4")
            return candidate

    def _decode_with_cv2(self, video_source, start: int, end: int):
        if cv2 is None:
            raise RuntimeError("Neither torchcodec nor cv2 is available for video decoding")
        temp_path = None
        try:
            if isinstance(video_source, BytesIO):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(video_source.getvalue())
                    temp_path = tmp.name
                video_path = temp_path
            else:
                video_path = video_source

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            current = start
            while True:
                if end is not None and current >= end:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                current += 1
            cap.release()
            if not frames:
                raise RuntimeError(f"No frames decoded from: {video_path}")
            frames_array = np.stack(frames)
            return torch.from_numpy(frames_array).float()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def __getitem__(self, idx):
        item = self.entries[idx]
        vid_name = item["vid_name"]
        start = int(item.get("start_frame", 0))
        end = int(item.get("end_frame", None))
        label = int(item.get("label", 0))

        cache_path = self._get_cache_path(vid_name, start, end)
        frames = self._load_from_cache(cache_path)

        if frames is None:
            video_source = self._open_video_bytes(vid_name)
            if VideoDecoder is not None:
                decoder = VideoDecoder(video_source)
                decoded_frames = decoder[start:end]
                if isinstance(decoded_frames, torch.Tensor):
                    frames = decoded_frames.float()
                else:
                    if isinstance(decoded_frames, np.ndarray):
                        frames = torch.from_numpy(decoded_frames).float()
                    else:
                        frames = torch.from_numpy(np.asarray(decoded_frames)).float()
            else:
                frames = self._decode_with_cv2(video_source, start, end)

            frames = self._uniform_sample_frames(frames)
            if isinstance(frames, torch.Tensor):
                self._save_to_cache(cache_path, frames.numpy())
            else:
                self._save_to_cache(cache_path, frames)

        frames = frames / 255.0 if frames.max() > 1.0 else frames
        frames = self._normalize_frames(frames)
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, torch.tensor(label, dtype=torch.long)


def create_loaders(data_config, batch_size=16, num_workers=4, pin_memory=True):
    train_dataset = DivingDataset(data_config, split="train")
    test_dataset = DivingDataset(data_config, split="test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             prefetch_factor=2, persistent_workers=True)
    return train_loader, test_loader


class BarGraphVisualizer:
    def __init__(self, categories=None, figsize=(10, 6)):
        self.categories = categories or []
        self.figsize = figsize

    def plot_training_history(self, epochs, train_losses, test_accuracies=None, title="Training History", save_path=None):
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=self.figsize)
        color_loss = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss", color=color_loss)
        ax1.plot(epochs, train_losses, color=color_loss, marker="o", label="Train Loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)
        if test_accuracies:
            ax2 = ax1.twinx()
            color_acc = "tab:green"
            ax2.set_ylabel("Test Accuracy", color=color_acc)
            ax2.plot(epochs, test_accuracies, color=color_acc, marker="s", label="Test Accuracy")
            ax2.tick_params(axis="y", labelcolor=color_acc)
            ax2.set_ylim([0, 1.0])
        ax1.set_title(title)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def _prepare_batch(images):
    import numpy as _np
    if isinstance(images, torch.Tensor):
        if images.ndim == 4:
            if images.shape[1] in (1, 3):
                return images.float()
            elif images.shape[-1] in (1, 3):
                imgs = images.permute(0, 3, 1, 2).float()
                return imgs
        elif images.ndim == 5:
            if images.shape[-1] in (1, 3):
                return images.permute(0, 1, 4, 2, 3).float()
            elif images.shape[2] in (1, 3):
                return images.float()
    if isinstance(images, (list, tuple)):
        processed = []
        for s in images:
            if not isinstance(s, torch.Tensor):
                if isinstance(s, _np.ndarray):
                    s = torch.from_numpy(s).float()
                else:
                    s = torch.from_numpy(_np.asarray(s)).float()
            if s.ndim == 4:
                if s.shape[-1] in (1, 3):
                    frame = s.permute(0, 3, 1, 2).float()
                elif s.shape[1] in (1, 3):
                    frame = s.float()
                else:
                    raise ValueError(f"Unsupported clip shape: {s.shape}")
            elif s.ndim == 3:
                frame = s.permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unsupported sample shape: {s.shape}")
            processed.append(frame)
        first = processed[0]
        if first.ndim == 3:
            return torch.stack(processed, dim=0)
        return torch.stack(processed, dim=0)
    raise ValueError("Unsupported batch images format")


def _evaluate_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            imgs = _prepare_batch(images)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return (correct / total) if total > 0 else 0.0


def train_model(
    model,
    train_loader,
    test_loader,
    device=None,
    num_epochs=30,
    visualize_history=True,
    history_save_path=None,
    learning_rate=0.01,
    gradient_accumulation_steps=1,
    use_mixed_precision=True,
    weight_decay=1e-4,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = GradScaler() if use_mixed_precision else None
    criterion = nn.CrossEntropyLoss()
    visualizer = BarGraphVisualizer()

    epochs = []
    train_losses = []
    test_accuracies = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (images, labels) in enumerate(train_loader):
            imgs = _prepare_batch(images)
            imgs, labels = imgs.to(device), labels.to(device)
            if use_mixed_precision:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            running_loss += loss.item() * imgs.size(0) * gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        scheduler.step()
        denom = len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1
        epoch_loss = running_loss / denom
        epochs.append(epoch + 1)
        train_losses.append(epoch_loss)
        if test_loader is not None:
            model.eval()
            test_accuracy = _evaluate_accuracy(model, test_loader, device)
            test_accuracies.append(test_accuracy)
            model.train()
    model.eval()
    if visualize_history and epochs:
        save_path = history_save_path or os.path.join("results", "training_history.png")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        visualizer.plot_training_history(epochs=epochs, train_losses=train_losses, test_accuracies=test_accuracies if test_accuracies else None, save_path=save_path)
    return model


def _load_model_from_subfolder(model_name: str, cfg_obj: DivingConfig = None):
    model_dir = os.path.join("models", model_name)
    model_file = os.path.join(model_dir, "model.py")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    spec = importlib.util.spec_from_file_location(f"models.{model_name}.model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "build_model") and callable(mod.build_model):
        return mod.build_model(cfg_obj) if cfg_obj is not None else mod.build_model()
    if hasattr(mod, "get_model") and callable(mod.get_model):
        return mod.get_model(cfg_obj) if cfg_obj is not None else mod.get_model()
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                sig = inspect.signature(obj.__init__)
                params = list(sig.parameters.keys())[1:]
                if cfg_obj is not None:
                    try:
                        num_classes = cfg_obj.get_labels().get("categories", [])
                        num_classes = len(num_classes) if num_classes else cfg_obj.get_training().get("num_classes", None)
                    except Exception:
                        num_classes = None
                else:
                    num_classes = None
                if "in_channels" in params and "num_classes" in params:
                    try:
                        return obj(3, int(num_classes) if num_classes else 5)
                    except Exception:
                        pass
                try:
                    return obj()
                except Exception:
                    continue
        except Exception:
            continue
    raise RuntimeError(f"No suitable model class found in {model_file}")
