import torch
import torch.nn as nn
import rootutils
from torchvision import models

rootutils.setup_root(__file__, indicator="src", pythonpath=True)


class TileResNet(nn.Module):
    def __init__(self, num_classes: int = 10, noise_sigma: float = 0.0):
        super().__init__()
        self.noise_sigma = noise_sigma
        r = models.resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4

        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(2048, num_classes)

        self.simkin_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.simkin_max = nn.AdaptiveMaxPool2d((1, 1))
        self.simkin_head = nn.Linear(512, 1)               # 256 avg + 256 max

    def forward(self, x, mode: str = 'clas'):
        if mode == 'simk':
            h = self.layer1(self.stem(x))                  # [B,256,H,W]
            a = self.simkin_avg(h).flatten(1)
            m = self.simkin_max(h).flatten(1)
            return self.simkin_head(torch.cat([a, m], dim=1))

        x = self.layer1(self.stem(x))
        
        if self.noise_sigma > 0: 
            x = x + torch.randn_like(x) * self.noise_sigma
        
        x = self.layer4(self.layer3(self.layer2(x)))
        return self.class_head(self.class_pool(x).flatten(1))
