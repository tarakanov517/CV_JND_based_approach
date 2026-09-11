from collections import OrderedDict

import torch
from torch import nn

from noises import (
    ContrastAdaptiveNoise,
    LateralInhibitionNoise,
    NoisyConv2d,
    PyramidalHeadNoise,
)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class CORblockRT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        out_shape=None,
        activation_noise=None,
        sigma_axon=0.0,
        sigma_dendrite=0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.conv_input = NoisyConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            sigma_axon=sigma_axon,
            sigma_dendrite=sigma_dendrite,
        )
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.activation_noise = activation_noise or nn.Identity()
        self.nonlin_input = nn.ReLU(inplace=True)
        self.conv1 = NoisyConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            sigma_axon=sigma_axon,
            sigma_dendrite=sigma_dendrite,
        )
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

    def forward(self, inputs=None, state=None, batch_size=None, device=None):
        if inputs is None:
            inputs = torch.zeros(
                batch_size,
                self.out_channels,
                self.out_shape,
                self.out_shape,
                device=device,
            )
        else:
            inputs = self.conv_input(inputs)
            inputs = self.norm_input(inputs)
            inputs = self.activation_noise(inputs)
            inputs = self.nonlin_input(inputs)
        if state is None:
            state = 0
        outputs = self.conv1(inputs + state)
        outputs = self.norm1(outputs)
        outputs = self.nonlin1(outputs)
        return outputs, outputs


class CORnetRT(nn.Module):
    def __init__(
        self,
        num_classes=257,
        times=5,
        sigma_lateral=0.0,
        sigma_prop=0.0,
        sigma_add=0.0,
        pyramid_enabled=False,
        sigma_pyramid=0.0,
        gamma=1.0,
        b=1.0,
        sigma_axon=0.0,
        sigma_dendrite=0.0,
    ):
        super().__init__()
        self.times = times
        parameter_noise = {
            "sigma_axon": sigma_axon,
            "sigma_dendrite": sigma_dendrite,
        }
        self.V1 = CORblockRT(
            3,
            64,
            kernel_size=7,
            stride=4,
            out_shape=56,
            activation_noise=LateralInhibitionNoise(sigma_lateral),
            **parameter_noise,
        )
        self.V2 = CORblockRT(
            64,
            128,
            stride=2,
            out_shape=28,
            activation_noise=ContrastAdaptiveNoise(sigma_prop, sigma_add),
            **parameter_noise,
        )
        self.V4 = CORblockRT(128, 256, stride=2, out_shape=14, **parameter_noise)
        self.IT = CORblockRT(256, 512, stride=2, out_shape=7, **parameter_noise)
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("avgpool", nn.AdaptiveAvgPool2d(1)),
                    ("flatten", Flatten()),
                    (
                        "pyramidal",
                        PyramidalHeadNoise(
                            pyramid_enabled, sigma_pyramid, gamma, b
                        ),
                    ),
                    ("linear", nn.Linear(512, num_classes)),
                ]
            )
        )

    def forward(self, inputs):
        outputs = {"inp": inputs}
        states = {}
        blocks = ["inp", "V1", "V2", "V4", "IT"]
        for block in blocks[1:]:
            block_input = outputs["inp"] if block == "V1" else None
            output, state = getattr(self, block)(
                block_input,
                batch_size=inputs.size(0),
                device=inputs.device,
            )
            outputs[block] = output
            states[block] = state
        for _ in range(1, self.times):
            new_outputs = {"inp": inputs}
            for index, block in enumerate(blocks[1:], start=1):
                output, state = getattr(self, block)(
                    outputs[blocks[index - 1]], states[block]
                )
                new_outputs[block] = output
                states[block] = state
            outputs = new_outputs
        return self.decoder(outputs["IT"])


class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, inputs):
        return self.model((inputs - self.mean) / self.std)
