"""
PyTorch dataset wrapper for the handwriting character dataset.

Loads .npz files written by data/synthesize.py or data/import_custom.py and
exposes them as a torch Dataset. Optional on-the-fly augmentation can be
applied during training so the model sees a slightly different version of
each sample every epoch.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentationPipeline


class HandwritingDataset(Dataset):
    def __init__(self, npz_path: str, augment: Optional[AugmentationPipeline] = None,
                 normalize: bool = True):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found, run python -m data.synthesize first")

        data = np.load(path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])

        if self.augment is not None:
            img = self.augment(img)

        tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1, 28, 28]
        if self.normalize:
            tensor = tensor / 255.0
            tensor = (tensor - 0.5) / 0.5  # [-1, 1]

        return tensor, label


def load_classes(npz_dir: str):
    """Read the classes.txt that synthesize.py writes alongside the npz files."""
    classes_path = Path(npz_dir) / "classes.txt"
    with open(classes_path) as f:
        return [line.strip() for line in f if line.strip()]
