import torch
import copy

from torchvision import models
from torch import nn


class DualBN(nn.Module):
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
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1,
            resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        
        convert_bn_to_dual(self.stem)
        convert_bn_to_dual(self.layer1)
        
        
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Classification Head
        self.class_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(2048, num_classes)
        
        # Simkin Head: avg + max пулинг, чтобы локальный пик возмущения сохранялся
        self.simkin_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.simkin_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.simkin_head = nn.Linear(512, 1)   # 256 (avg) + 256 (max)

    def set_domain(self, d):
        for m in self.modules():
            if isinstance(m, DualBN):
                m.domain = d

    def forward(self, x, mode='clas'):
        if mode == 'simk':
            self.set_domain(mode)
            feat1, feat2 = x                       # два патча: левое и правое поле
            feat1 = self.layer1(self.stem(feat1))
            feat2 = self.layer1(self.stem(feat2))
            diff = torch.abs(feat2 - feat1)
            avg = self.simkin_avgpool(diff).flatten(1)
            mx  = self.simkin_maxpool(diff).flatten(1)
            h = torch.cat([avg, mx], dim=1)        # [B, 512]
            return self.simkin_head(h)
        
        self.set_domain(mode)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.class_pool(x).flatten(1)
        return self.class_head(x)
        