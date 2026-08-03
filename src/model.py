import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibitionNoise(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = float(sigma)
        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, inputs):
        if not self.training or self.sigma == 0:
            return inputs

        white_noise = torch.randn_like(inputs)
        channels = inputs.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        correlated_noise = F.conv2d(
            white_noise,
            kernel,
            padding=1,
            groups=channels,
        )
        noise_std = correlated_noise.std(unbiased=False).clamp_min(1e-8)
        normalized_noise = correlated_noise / noise_std
        return inputs + self.sigma * normalized_noise


class ContrastAdaptiveNoise(nn.Module):
    def __init__(self, sigma_prop=0.0, sigma_add=0.0):
        super().__init__()
        self.sigma_prop = float(sigma_prop)
        self.sigma_add = float(sigma_add)

    def forward(self, inputs):
        if not self.training or (self.sigma_prop == 0 and self.sigma_add == 0):
            return inputs

        signal_magnitude = inputs.abs() + 1e-6
        variance = (self.sigma_prop * signal_magnitude).square()
        variance = variance + self.sigma_add**2
        noise = torch.randn_like(inputs) * variance.sqrt()
        return inputs + noise


class PyramidalHeadNoise(nn.Module):
    def __init__(self, sigma=0.0, gamma=1.0, b=1.0):
        super().__init__()
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.b = float(b)

    def forward(self, inputs):
        signal = inputs.abs().pow(self.gamma)
        inhibition = signal.mean(dim=1, keepdim=True)
        outputs = inputs.sign() * signal / (self.b + inhibition)

        if self.training and self.sigma != 0:
            outputs = outputs + torch.randn_like(outputs) * self.sigma

        return outputs


class Net(nn.Module):
    def __init__(
        self,
        sigma_lateral=0.0,
        sigma_prop=0.0,
        sigma_add=0.0,
        sigma=0.0,
        gamma=1.0,
        b=1.0,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            LateralInhibitionNoise(sigma_lateral),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            ContrastAdaptiveNoise(sigma_prop=sigma_prop, sigma_add=sigma_add),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            PyramidalHeadNoise(sigma=sigma, gamma=gamma, b=b),
            nn.Linear(256, 10),
        )

    def forward(self, inputs):
        outputs = self.block1(inputs)
        outputs = self.block2(outputs)
        return self.classifier(outputs)
