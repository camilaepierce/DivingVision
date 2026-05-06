
# from tqdm import tqdm
from io import BytesIO
import hashlib
import json
import os
import pickle
import tarfile
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None

try:
    import cv2
except Exception:
    cv2 = None


class DivingDataset(Dataset):
    """PyTorch Dataset for Diving48-style JSON splits.

    Expects `data_config` dict with keys:
      - "rgb_data": path to directory of extracted videos or path to a tar archive
      - "train_split": path to train json
      - "test_split": path to test json

    Each json entry must contain: `vid_name`, `start_frame`, `end_frame`, `label`.
    
    Features:
            - Uniform sampling of 16 frames per sample
            - Frame resizing to 224x224 for efficiency
      - Disk caching to avoid re-decoding
      - ImageNet normalization (can be disabled with normalize=False)
    """

    # ImageNet normalization constants
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
        
        # Setup caching directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(self.rgb_source or "."), ".frame_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # decide whether rgb_source is a tar archive or a directory
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
        """Generate cache file path for video frames."""
        key = f"{vid_name}_{start}_{end}_{self.frame_size}_{self.num_frames}"
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")

    def _uniform_sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Uniformly sample self.num_frames frames across the clip.

        If the clip is shorter than the target length, frame indices are
        repeated to preserve a fixed output size.
        """
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
        """Load frames from disk cache."""
        try:
            with open(cache_path, "rb") as f:
                frames = pickle.load(f)
            return torch.from_numpy(frames).float()
        except Exception:
            return None

    def _save_to_cache(self, cache_path: str, frames: np.ndarray):
        """Save frames to disk cache."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(frames, f)
        except Exception:
            pass  # Silently fail if cache write fails

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to frames.
        Assumes frames are in [0, 1] range.
        """
        if not self.normalize:
            return frames
        
        # frames shape: (T, H, W, C) or (T, C, H, W)
        mean = torch.tensor(self.IMAGENET_MEAN, device=frames.device, dtype=frames.dtype)
        std = torch.tensor(self.IMAGENET_STD, device=frames.device, dtype=frames.dtype)
        
        # Reshape for broadcasting
        if frames.ndim == 4:
            if frames.shape[-1] == 3:
                # (T, H, W, C)
                mean = mean.reshape(1, 1, 1, 3)
                std = std.reshape(1, 1, 1, 3)
            else:
                # (T, C, H, W)
                mean = mean.reshape(1, 3, 1, 1)
                std = std.reshape(1, 3, 1, 1)
        elif frames.ndim == 3:
            if frames.shape[-1] == 3:
                # (H, W, C)
                mean = mean.reshape(1, 1, 3)
                std = std.reshape(1, 1, 3)
            else:
                # (C, H, W)
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
                # try without rgb/ prefix
                member = self._tar.getmember(f"{vid_name}.mp4")
            fobj = self._tar.extractfile(member)
            return BytesIO(fobj.read())
        else:
            # assume rgb_source is a directory containing either files under rgb/ or directly
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
                # Resize frame to 224x224 for efficiency
                frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                current += 1

            cap.release()

            if not frames:
                raise RuntimeError(f"No frames decoded from: {video_path}")

            # Convert list of numpy arrays to single array first, then to tensor (much faster)
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

        # Check cache first
        cache_path = self._get_cache_path(vid_name, start, end)
        frames = self._load_from_cache(cache_path)
        
        if frames is None:
            # Not in cache, decode video
            video_source = self._open_video_bytes(vid_name)

            if VideoDecoder is not None:
                # VideoDecoder accepts a path or a file-like object / bytes buffer
                decoder = VideoDecoder(video_source)
                # slice frames
                decoded_frames = decoder[start:end]

                # ensure tensor type and float32
                if isinstance(decoded_frames, torch.Tensor):
                    frames = decoded_frames.float()
                else:
                    # Convert numpy array to tensor efficiently
                    if isinstance(decoded_frames, np.ndarray):
                        frames = torch.from_numpy(decoded_frames).float()
                    else:
                        frames = torch.from_numpy(np.asarray(decoded_frames)).float()
            else:
                frames = self._decode_with_cv2(video_source, start, end)

            # Ensure a fixed number of uniformly sampled frames per sample.
            frames = self._uniform_sample_frames(frames)
            
            # Save to cache for future use
            if isinstance(frames, torch.Tensor):
                self._save_to_cache(cache_path, frames.numpy())
            else:
                self._save_to_cache(cache_path, frames)

        # Normalize frames to [0, 1] range if needed
        frames = frames / 255.0 if frames.max() > 1.0 else frames
        
        # Apply ImageNet normalization
        frames = self._normalize_frames(frames)
        
        # optional transform per-clip
        if self.transform is not None:
            frames = self.transform(frames)
        
        # Ensure frames are in (T, C, H, W) format for PyTorch Conv3d
        # frames should be (T, H, W, C) after decoding, normalize it if needed
        if frames.ndim == 4 and frames.shape[-1] == 3:
            # Convert from (T, H, W, C) to (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2).contiguous()

        return frames, torch.tensor(label, dtype=torch.long)
    


