"""
CNN architecture for 62-class handwriting character recognition.

The network is intentionally small. The full dataset is heavy augmentation on
top of a synthetic base, so the bottleneck is not capacity, it is regularization
under varied inputs. A wider deeper net just overfits the augmentation pattern.

Design choices:
  - 3 conv blocks, 32 -> 64 -> 128 channels
  - BatchNorm after every conv for stable training without learning rate tuning
  - Dropout in the classifier head to prevent overfit on per-font features
  - AdaptiveAvgPool so any 28-ish input size works at inference time
"""

import torch
import torch.nn as nn


class HandwritingCNN(nn.Module):
    def __init__(self, num_classes: int = 62, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28 -> 14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 14 -> 7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 7 -> 3 after pool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HandwritingCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"output shape: {out.shape}")
    print(f"params: {model.count_params():,}")
