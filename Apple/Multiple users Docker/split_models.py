from __future__ import annotations

from torch import nn
import torch


class SplitClientNet(nn.Module):
    def __init__(self, cut_layer: int) -> None:
        super().__init__()
        if cut_layer not in (0, 1):
            raise ValueError("cut_layer must be 0 or 1")
        self.cut_layer = cut_layer
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        if self.cut_layer == 0:
            return x
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        return x


class SplitServerNet(nn.Module):
    def __init__(self, cut_layer: int) -> None:
        super().__init__()
        if cut_layer not in (0, 1):
            raise ValueError("cut_layer must be 0 or 1")
        self.cut_layer = cut_layer
        self.act = nn.Tanh()
        if cut_layer == 0:
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cut_layer == 0:
            x = self.act(self.conv2(x))
            x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
