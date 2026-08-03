import torch
import torch.nn as nn
import rootutils
from torchvision import models

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

_TAPS = ['conv1', 'stem', 'l1.0', 'l1.1', 'l1', 'l2', 'l3', 'l4']
_CH = {'conv1': 64, 'stem': 64, 'l1.0': 256, 'l1.1': 256, 'l1': 256,
       'l2': 512, 'l3': 1024, 'l4': 2048}


class TileResNet(nn.Module):

    def __init__(self, num_classes: int = 10, noise_sigma: float = 0.0, tap: str = 'l1'):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.tap = tap
        r = models.resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1 
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4

        self._stages = [
            ('conv1', self.stem[:3]), ('stem', self.stem[3]),
            ('l1.0', self.layer1[0]), ('l1.1', self.layer1[1]), ('l1', self.layer1[2]),
            ('l2', self.layer2), ('l3', self.layer3), ('l4', self.layer4),
        ]
        self._tap_idx = {n: i for i, (n, _) in enumerate(self._stages)}[tap]

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
        f = self._frontend(x)
        if mode == 'simk':
            a = self.simkin_avg(f).flatten(1)
            m = self.simkin_max(f).flatten(1)
            return self.simkin_head(torch.cat([a, m], dim=1))

        if self.noise_sigma > 0:                                      # шум после фронт-энда
            f = f + torch.randn_like(f) * self.noise_sigma
        x = self._backend(f)
        return self.class_head(self.class_pool(x).flatten(1))
