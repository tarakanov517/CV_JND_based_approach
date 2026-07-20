import math

import torch
import torch.nn.functional as functional
from torch import nn


class LateralInhibitionNoise(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = float(sigma)
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        self.register_buffer("kernel", kernel[None, None])

    def forward(self, inputs):
        if not self.training or self.sigma <= 0:
            return inputs
        channel_count = inputs.size(1)
        kernel = self.kernel.expand(channel_count, 1, 3, 3)
        noise = functional.conv2d(
            torch.randn_like(inputs),
            kernel,
            padding=1,
            groups=channel_count,
        )
        scale = noise.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-8)
        return inputs + self.sigma * noise / scale


class ContrastAdaptiveNoise(nn.Module):
    def __init__(self, sigma_proportional=0.0, sigma_additive=0.0):
        super().__init__()
        self.sigma_proportional = float(sigma_proportional)
        self.sigma_additive = float(sigma_additive)

    def forward(self, inputs):
        if not self.training or (
            self.sigma_proportional <= 0 and self.sigma_additive <= 0
        ):
            return inputs
        variance = (
            self.sigma_proportional * inputs.detach().abs()
        ).square() + self.sigma_additive**2
        return inputs + variance.sqrt() * torch.randn_like(inputs)


class MagnoParvoNoise(nn.Module):
    def __init__(
        self,
        ratio_magno=0.1,
        sigma_magno=0.0,
        fano_magno=1.5,
        sigma_parvo=0.0,
        fano_parvo=1.1,
    ):
        super().__init__()
        self.ratio_magno = float(ratio_magno)
        self.sigma_magno = float(sigma_magno)
        self.fano_magno = float(fano_magno)
        self.sigma_parvo = float(sigma_parvo)
        self.fano_parvo = float(fano_parvo)

    def forward(self, inputs):
        if not self.training or (self.sigma_magno <= 0 and self.sigma_parvo <= 0):
            return inputs
        channel_count = inputs.size(1)
        magno_count = max(
            1,
            min(channel_count - 1, round(channel_count * self.ratio_magno)),
        )
        outputs = inputs.clone()
        outputs[:, :magno_count] += (
            self.sigma_magno
            * math.sqrt(self.fano_magno)
            * torch.randn_like(outputs[:, :magno_count])
        )
        outputs[:, magno_count:] += (
            self.sigma_parvo
            * math.sqrt(self.fano_parvo)
            * torch.randn_like(outputs[:, magno_count:])
        )
        return outputs


class AxonalNoise(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, inputs):
        if not self.training or self.sigma <= 0:
            return inputs
        return inputs + self.sigma * torch.randn_like(inputs)


class DendriticOUNoise(nn.Module):
    def __init__(self, theta=1.0, sigma=0.0, mu=0.0, dt=1.0):
        super().__init__()
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.dt = float(dt)
        self.register_buffer("state", torch.empty(0), persistent=False)

    def forward(self, inputs):
        if not self.training or self.sigma <= 0:
            return inputs
        state_shape = (1, *inputs.shape[1:])
        if self.state.shape != state_shape or self.state.device != inputs.device:
            self.state = torch.full(state_shape, self.mu, device=inputs.device, dtype=inputs.dtype)
        diffusion = self.sigma * math.sqrt(self.dt) * torch.randn_like(self.state)
        self.state = (
            self.state
            + self.theta * (self.mu - self.state) * self.dt
            + diffusion
        ).detach()
        return inputs + self.state


class PyramidalNoise(nn.Module):
    def __init__(
        self,
        enabled=False,
        sigma=0.0,
        gamma=1.0,
        semisaturation=1.0,
        gain_decay=0.1,
        inhibition_size=3,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.semisaturation = float(semisaturation)
        self.gain_decay = float(gain_decay)
        self.inhibition_size = int(inhibition_size)

    def forward(self, inputs):
        if not self.enabled:
            return inputs
        magnitude = inputs.abs().pow(self.gamma)
        local_activity = functional.avg_pool2d(
            magnitude,
            kernel_size=self.inhibition_size,
            stride=1,
            padding=self.inhibition_size // 2,
        )
        gain = 1.0 / (1.0 + self.gain_decay * inputs.abs())
        outputs = gain * inputs.sign() * magnitude
        outputs = outputs / (self.semisaturation + local_activity)
        if self.training and self.sigma > 0:
            outputs = outputs + self.sigma * torch.randn_like(outputs)
        return outputs


class EarlyVisualNoise(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lateral = LateralInhibitionNoise(config.get("lateral_sigma", 0.0))
        self.contrast = ContrastAdaptiveNoise(
            config.get("contrast_sigma_proportional", 0.0),
            config.get("contrast_sigma_additive", 0.0),
        )
        self.magno_parvo = MagnoParvoNoise(
            ratio_magno=config.get("ratio_magno", 0.1),
            sigma_magno=config.get("sigma_magno", 0.0),
            fano_magno=config.get("fano_magno", 1.5),
            sigma_parvo=config.get("sigma_parvo", 0.0),
            fano_parvo=config.get("fano_parvo", 1.1),
        )

    def forward(self, inputs):
        outputs = self.lateral(inputs)
        outputs = self.contrast(outputs)
        return self.magno_parvo(outputs)


class NoisyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config, index):
        super().__init__()
        dendrite_theta = config.get("dendrite_theta", (1.0, 1.0, 1.0))[index]
        dendrite_sigma = config.get("dendrite_sigma", (0.0, 0.0, 0.0))[index]
        axon_sigma = config.get("axon_sigma", (0.0, 0.0, 0.0))[index]
        self.dendrite1 = DendriticOUNoise(dendrite_theta, dendrite_sigma)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.axon1 = AxonalNoise(axon_sigma)
        self.dendrite2 = DendriticOUNoise(dendrite_theta, dendrite_sigma)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.axon2 = AxonalNoise(axon_sigma)
        self.visual_noise = EarlyVisualNoise(config) if index == 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        outputs = self.dendrite1(inputs)
        outputs = self.axon1(self.relu1(self.norm1(self.conv1(outputs))))
        outputs = self.dendrite2(outputs)
        outputs = self.axon2(self.relu2(self.norm2(self.conv2(outputs))))
        return self.pool(self.visual_noise(outputs))


class Net(nn.Module):
    def __init__(self, noise_config=None):
        super().__init__()
        config = noise_config or {}
        dendrite_theta = config.get("dendrite_theta", (1.0, 1.0, 1.0))
        dendrite_sigma = config.get("dendrite_sigma", (0.0, 0.0, 0.0))
        axon_sigma = config.get("axon_sigma", (0.0, 0.0, 0.0))
        self.block1 = NoisyConvBlock(3, 32, config, 0)
        self.block2 = NoisyConvBlock(32, 64, config, 1)
        self.pyramidal = PyramidalNoise(
            enabled=config.get("pyramidal_enabled", False),
            sigma=config.get("pyramidal_sigma", 0.0),
            gamma=config.get("pyramidal_gamma", 1.0),
            semisaturation=config.get("pyramidal_semisaturation", 1.0),
            gain_decay=config.get("pyramidal_gain_decay", 0.1),
        )
        self.flatten = nn.Flatten()
        self.dendrite = DendriticOUNoise(dendrite_theta[2], dendrite_sigma[2])
        self.linear1 = nn.Linear(8 * 8 * 64, 256)
        self.relu = nn.ReLU()
        self.axon = AxonalNoise(axon_sigma[2])
        self.linear2 = nn.Linear(256, 10)

    def forward(self, inputs):
        outputs = self.block1(inputs)
        outputs = self.block2(outputs)
        outputs = self.pyramidal(outputs)
        outputs = self.dendrite(self.flatten(outputs))
        outputs = self.axon(self.relu(self.linear1(outputs)))
        return self.linear2(outputs)
