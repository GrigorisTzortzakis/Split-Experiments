import torch
import torch.nn as nn


class LeNetComplete(nn.Module):
    """
    Caffe-style MNIST LeNet (common in Caffe-based implementations):
      conv(1 -> 20, 5x5) -> ReLU -> MaxPool(2,2)
      conv(20 -> 50, 5x5) -> ReLU -> MaxPool(2,2)
      FC(50*4*4=800 -> 500) -> ReLU
      FC(500 -> 10)
    Input: (N, 1, 28, 28)
    Output: logits (N, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=50 * 4 * 4, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Paper mentions Xavier/Gaussian-style init generally; this is a solid Xavier default.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)         # (N, 10)
        return x


class LeNetClientNetwork(nn.Module):
    """
    Client side (up to the split): first conv block only.
    Output shape for MNIST: (N, 20, 12, 12)
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block1(x)


class LeNetServerNetwork(nn.Module):
    """
    Server side (from the split to the end):
      conv(20 -> 50, 5x5) -> ReLU -> MaxPool
      FC(800 -> 500) -> ReLU
      FC(500 -> 10)
    Expects input shape: (N, 20, 12, 12)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Linear(in_features=50 * 4 * 4, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(x)              # (N, 50, 4, 4)
        x = x.view(x.size(0), -1)       # (N, 800)
        x = self.block3(x)              # (N, 10)
        return x


if __name__ == "__main__":
    # quick shape sanity check
    x = torch.randn(8, 1, 28, 28)
    net = LeNetComplete()
    y = net(x)
    print("LeNetComplete:", y.shape)

    client = LeNetClientNetwork()
    server = LeNetServerNetwork()
    h = client(x)
    y2 = server(h)
    print("Split:", h.shape, "->", y2.shape)