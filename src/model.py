import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianActivationNoise(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if not self.training or self.sigma == 0:
            return x

        noise = torch.randn_like(x)
        scale = x.detach().std()
        return x + self.sigma * scale * noise


class Net(nn.Module):
    def __init__(self, sigma1, sigma2, sigma3):
        super().__init__()
        self.block1 = nn.Sequential(  # 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),  # 32x32x32
            nn.ReLU(),  # 32x32x32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),  # 32x32x32
            nn.ReLU(),  # 32x32x32
            GaussianActivationNoise(sigma1),
            nn.MaxPool2d(2),  # 32x16x16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),  # 64x16x16
            nn.ReLU(),  # 64x16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),  # 64x16x16
            nn.ReLU(),  # 64x16x16
            GaussianActivationNoise(sigma2),
            nn.MaxPool2d(2),  # 64x8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            GaussianActivationNoise(sigma3),
            nn.Linear(256, 10),
        )

    def set_activation_sigmas(self, sigma1, sigma2, sigma3):
        self.block1[6].sigma = float(sigma1)
        self.block2[6].sigma = float(sigma2)
        self.classifier[3].sigma = float(sigma3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)

        return x
