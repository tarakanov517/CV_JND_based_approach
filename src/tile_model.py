import torch
import torch.nn as nn
import rootutils
from torchvision import models

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

_TAPS = ['conv1', 'stem', 'l1.0', 'l1.1', 'l1', 'l2', 'l3', 'l4']
_CH = {'conv1': 64, 'stem': 64, 'l1.0': 256, 'l1.1': 256, 'l1': 256,
       'l2': 512, 'l3': 1024, 'l4': 2048}


class TileResNet(nn.Module):

    def __init__(self, num_classes: int = 10, noise_sigma: float = 0.0, tap: str = 'l1',
                 noise_tap: str = None):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.tap = tap                       # ветвление Симкин-головы
        self.noise_tap = noise_tap or tap    # место впрыскивания шума
        r = models.resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1 
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4

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

    def forward(self, x, mode: str = 'clas'):
        if mode == 'simk':                                # голова ветвится на tap
            f = self._frontend(x)
            a = self.simkin_avg(f).flatten(1)
            m = self.simkin_max(f).flatten(1)
            return self.simkin_head(torch.cat([a, m], dim=1))

        for i, (_, mod) in enumerate(self._stages):       # шум впрыскивается на noise_tap
            x = mod(x)
            if i == self._noise_idx and self.noise_sigma > 0:
                x = x + torch.randn_like(x) * self.noise_sigma
        return self.class_head(self.class_pool(x).flatten(1))
