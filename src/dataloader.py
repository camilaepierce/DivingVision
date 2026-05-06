import torch
from torch.utils.data import DataLoader
from .dataset import DivingDataset


def create_loaders(data_config, batch_size=16, num_workers=4, pin_memory=None):
    """Create train and test PyTorch DataLoaders from data_config dict.

    Args:
        data_config: Configuration dictionary with data paths
        batch_size: Batch size (default 16 for better GPU utilization)
        num_workers: Number of workers for async loading (default 4)
        pin_memory: Pin memory for faster GPU transfer (default: auto)

    Returns: train_loader, test_loader
    """
    train_dataset = DivingDataset(data_config, split="train")
    test_dataset = DivingDataset(data_config, split="test")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": None,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        **loader_kwargs,
    )

    return train_loader, test_loader
