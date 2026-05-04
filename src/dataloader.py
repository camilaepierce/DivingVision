from torch.utils.data import DataLoader
from .dataset import DivingDataset


def create_loaders(data_config, batch_size=16, num_workers=4, pin_memory=True):
    """Create train and test PyTorch DataLoaders from data_config dict.

    Args:
        data_config: Configuration dictionary with data paths
        batch_size: Batch size (default 16 for better GPU utilization)
        num_workers: Number of workers for async loading (default 4)
        pin_memory: Pin memory for faster GPU transfer (default True)

    Returns: train_loader, test_loader
    """
    train_dataset = DivingDataset(data_config, split="train")
    test_dataset = DivingDataset(data_config, split="test")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=None,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=None,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader, test_loader
