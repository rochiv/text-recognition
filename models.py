import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
        )

        self.adap_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adap_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
