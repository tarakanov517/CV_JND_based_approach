import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibitionNoise(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = float(sigma)
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer(
            "kernel", kernel.unsqueeze(0).unsqueeze(0), persistent=False
        )

    def forward(self, inputs):
        if not self.training or self.sigma == 0:
            return inputs
        white_noise = torch.randn_like(inputs)
        channels = inputs.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        correlated_noise = F.conv2d(white_noise, kernel, padding=1, groups=channels)
        noise_std = correlated_noise.std(unbiased=False).clamp_min(1e-8)
        return inputs + self.sigma * correlated_noise / noise_std


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
        return inputs + torch.randn_like(inputs) * variance.sqrt()


class PyramidalHeadNoise(nn.Module):
    def __init__(self, enabled=False, sigma=0.0, gamma=1.0, b=1.0):
        super().__init__()
        self.enabled = bool(enabled)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.b = float(b)

    def forward(self, inputs):
        if not self.enabled:
            return inputs
        signal = inputs.abs().pow(self.gamma)
        inhibition = signal.mean(dim=1, keepdim=True)
        outputs = inputs.sign() * signal / (self.b + inhibition)
        if self.training and self.sigma != 0:
            outputs = outputs + torch.randn_like(outputs) * self.sigma
        return outputs


class NoisyConv2d(nn.Conv2d):
    def __init__(self, *args, sigma_axon=0.0, sigma_dendrite=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_axon = float(sigma_axon)
        self.sigma_dendrite = float(sigma_dendrite)

    def forward(self, inputs):
        weight = self.weight
        if self.training:
            if self.sigma_axon != 0:
                weight = weight * (
                    1.0 + self.sigma_axon * torch.randn_like(weight)
                )
            if self.sigma_dendrite != 0:
                scale = self.weight.detach().std(unbiased=False).clamp_min(1e-8)
                weight = weight + self.sigma_dendrite * scale * torch.randn_like(weight)
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
