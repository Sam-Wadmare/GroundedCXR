"""
TASK 3: A convolutional neural network baseline trained FROM SCRATCH
(no pretrained weights), for comparison against the project's
ImageNet-pretrained ResNet18 (src/model.py).

Architecture: 4 blocks of [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d]
with doubling channel counts, followed by global average pooling and a
single linear classifier head. Input is expected to be
(batch, 3, 224, 224), matching the rest of the project's preprocessing
(src/inference.py, src/data_loader.py).

Usage (as a module):
    from src.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=2)

Usage (quick shape/parameter-count sanity check):
    python -m src.simple_cnn
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """One Conv -> BatchNorm -> ReLU -> MaxPool block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SimpleCNN(nn.Module):
    """
    4 Conv-BatchNorm-ReLU-MaxPool blocks (channels: 3 -> 32 -> 64 -> 128
    -> 256), then global average pooling collapses each feature map down
    to one number per channel, and a linear layer maps those 256 numbers
    to `num_classes` logits.

    No pretrained weights anywhere in this model — every parameter
    starts randomly initialized (PyTorch's default init), matching TASK
    3's requirement that this be a genuinely from-scratch baseline, not
    transfer learning.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        # Global average pooling: collapses each (H, W) feature map to a
        # single value per channel, regardless of input spatial size.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)   # 224 -> 112
        x = self.block2(x)   # 112 -> 56
        x = self.block3(x)   # 56  -> 28
        x = self.block4(x)   # 28  -> 14
        x = self.global_pool(x)   # (batch, 256, 1, 1)
        x = torch.flatten(x, 1)   # (batch, 256)
        x = self.classifier(x)    # (batch, num_classes)
        return x


if __name__ == "__main__":
    # Quick shape + parameter-count sanity check when run directly.
    model = SimpleCNN(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
