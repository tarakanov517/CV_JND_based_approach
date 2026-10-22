from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from torchvision import models

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.model import DualBN, convert_bn_to_dual   # раздельные BN для clas/simk

EPS = 1e-5
_CH = {'conv1': 64, 'stem': 64, 'l1.0': 256, 'l1.1': 256, 'l1': 256,
       'l2': 512, 'l3': 1024, 'l4': 2048}


class TileResNet(nn.Module):
    """ResNet50 с Симкин-головой на tap и инъекцией шума на noise_tap."""

    def __init__(self, num_classes: int = 10, noise_sigma: float = 0.0, tap: str = 'l1',
                 noise_tap: str = None, noise_type: Literal['neuronal', 'gaussian', None] = None,
                 dual_bn: bool = False, noise_scale: float = 0.35, noise_level: float = 0.07):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.tap = tap                       # ветвление Симкин-головы
        self.noise_tap = noise_tap or tap    # место впрыскивания шума
        self.set_noise_mode(noise_type, noise_scale, noise_level)
        self.dual_bn = dual_bn

        r = models.resnet50(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4
        if dual_bn:                          # раздельные BN до сборки stages, чтобы срезы их видели
            for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                convert_bn_to_dual(m)

        self._stages = [
            ('conv1', self.stem[:3]), ('stem', self.stem[3]),
            ('l1.0', self.layer1[0]), ('l1.1', self.layer1[1]), ('l1', self.layer1[2]),
            ('l2', self.layer2), ('l3', self.layer3), ('l4', self.layer4),
        ]
        _idx = {n: i for i, (n, _) in enumerate(self._stages)}
        self._tap_idx = _idx[tap]
        self._noise_idx = _idx[self.noise_tap]

        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(2048, num_classes)
        self.simkin_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.simkin_max = nn.AdaptiveMaxPool2d((1, 1))
        self.simkin_head = nn.Linear(2 * _CH[tap], 1)                 # avg + max

    def frontend_param_ids(self):
        ids = set()
        for _, m in self._stages[:self._tap_idx + 1]:
            ids.update(id(p) for p in m.parameters())
        return ids

    def _frontend(self, x):
        for _, m in self._stages[:self._tap_idx + 1]:
            x = m(x)
        return x

    def _backend(self, x):
        for _, m in self._stages[self._tap_idx + 1:]:
            x = m(x)
        return x

    def set_noise_mode(self, noise_type=None, noise_scale=1.0, noise_level=1.0):
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def set_domain(self, d):                 # переключить DualBN между clas/simk
        if self.dual_bn:
            for m in self.modules():
                if isinstance(m, DualBN):
                    m.domain = d

    def _inject_noise(self, x):
        if self.noise_type == 'neuronal':    # Poisson equivalent to Gauss, не зависит от noise_sigma
            u = self.noise_scale * x + self.noise_level
            u = u + torch.randn_like(u) * torch.sqrt(F.relu(u) + EPS)
            return (u - self.noise_level) / self.noise_scale
        if self.noise_sigma > 0:             # additive fixed σ (noise_type None или 'gaussian')
            return x + torch.randn_like(x) * self.noise_sigma
        return x

    def forward(self, x, mode: str = 'clas'):
        self.set_domain(mode)
        if mode == 'simk':                   # голова ветвится на tap
            f = self._frontend(x)
            a = self.simkin_avg(f).flatten(1)
            m = self.simkin_max(f).flatten(1)
            return self.simkin_head(torch.cat([a, m], dim=1))

        for i, (_, mod) in enumerate(self._stages):   # шум впрыскивается на noise_tap
            x = mod(x)
            if i == self._noise_idx:
                x = self._inject_noise(x)
        return self.class_head(self.class_pool(x).flatten(1))