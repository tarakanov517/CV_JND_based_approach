import torch

from torchvision import models
from torch import nn

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1,
            resnet.relu, resnet.maxpool
        )
        
        self.layer1 = resnet.layer1
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

    def forward(self, x, mode='clas'):
        x = self.stem(x)
        x = self.layer1(x)

        if mode == 'simk':
            avg = self.simkin_avgpool(x).flatten(1)
            mx = self.simkin_maxpool(x).flatten(1)
            x = torch.cat([avg, mx], dim=1)
            return self.simkin_head(x)
        
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.class_pool(x).flatten(1)
        return self.class_head(x)
        
        