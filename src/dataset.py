
# from tqdm import tqdm
from io import BytesIO

# import cv2
import numpy as np
# import PIL.Image
# from IPython.display import Image, clear_output, display
# import matplotlib.pyplot as plt
import random
# from scipy.signal import convolve2d

# PyTorch will be out main tool for playing with neural networks
import torch
# import torch.hub
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
from torchvision import models, datasets, transforms
from config import DivingConfig

# Define dataset for loading data
class DivingDataset(Dataset):
    
    def __init__(self, rgbData, flowData, labels):
        self.rgb = rgbData
        self.flow = flowData
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = normalize_image(self.images[idx])
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label = self.labels[idx]
        return img_tensor, label