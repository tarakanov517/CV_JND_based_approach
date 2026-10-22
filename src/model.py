import copy

import torch
from torch import nn
from torchvision import models


class DualBN(nn.Module):
    """BatchNorm с раздельной статистикой для доменов clas/simk."""

    def __init__(self, bn):
        super().__init__()
        self.bn_class = bn
        self.bn_simk = copy.deepcopy(bn)
        self.domain = "clas"

    def forward(self, x):
        return self.bn_class(x) if self.domain == "clas" else self.bn_simk(x)


def convert_bn_to_dual(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBN(child))
        else:
            convert_bn_to_dual(child)


class CustomResNet(nn.Module):
    """ResNet50 с сиамской Симкин-головой (два поля) и шумом после layer1."""

    def __init__(self, num_classes=10, noise_sigma=0.0) -> None:
        super().__init__()
        self.noise_sigma = noise_sigma
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        convert_bn_to_dual(self.stem)
        convert_bn_to_dual(self.layer1)
        self.layer2, self.layer3, self.layer4 = resnet.layer2, resnet.layer3, resnet.layer4

        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(2048, num_classes)
        # Simkin-голова: avg + max пулинг, чтобы локальный пик возмущения сохранялся
        self.simkin_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.simkin_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.simkin_head = nn.Linear(512, 1)   # 256 (avg) + 256 (max)

    def set_domain(self, d):
        for m in self.modules():
            if isinstance(m, DualBN):
                m.domain = d

    def forward(self, x, mode='clas'):
        self.set_domain(mode)
        if mode == 'simk':
            feat1, feat2 = x                       # два патча: левое и правое поле
            feat1 = self.layer1(self.stem(feat1))
            feat2 = self.layer1(self.stem(feat2))
            diff = torch.abs(feat2 - feat1)
            avg = self.simkin_avgpool(diff).flatten(1)
            mx = self.simkin_maxpool(diff).flatten(1)
            return self.simkin_head(torch.cat([avg, mx], dim=1))   # [B, 512]

        x = self.layer1(self.stem(x))
        if self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.class_head(self.class_pool(x).flatten(1))