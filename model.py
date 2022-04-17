import torch
import torch.nn as nn

class ModelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self._1_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )
        self._3_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self._4_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self._6_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self._7_hidden_layer = nn.Linear(in_features=128, out_features=64)
        self._8_hidden_layer = nn.Linear(in_features=64, out_features=32)
        self._9_hidden_layer = nn.Linear(in_features=32, out_features=7)

    def forward(self, item):
        item = self._1_hidden_layer(item)
        item = self._3_hidden_layer(item)
        item = self._4_hidden_layer(item)
        item = nn.functional.avg_pool2d(item, (16, 16))
        item = torch.squeeze(item)
        item = self._6_hidden_layer(item)
        item = self._7_hidden_layer(item)
        item = self._8_hidden_layer(item)
        item = self._9_hidden_layer(item)

        return item