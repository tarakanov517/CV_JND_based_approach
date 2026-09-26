"""VOneResNet50 под общий интерфейс (ImageNet-нормализованный вход, forward(x, mode='clas')).

Официальный чекпойнт (ImageNet): https://vonenetmodels.s3.us-east-2.amazonaws.com/vonenet_resnet50_e70.pth.tar
В нём {'flags': {...}, 'state_dict': {'module.vone_block...', 'module.bottleneck...', 'module.model...'}}.
Габоры (GFB.weight) лежат в state_dict, поэтому фильтры при загрузке совпадают с оригиналом.
"""
import torch
import torch.nn as nn

from src.VOne.vonenet import VOneNet

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ARCH_KEYS = ("stride", "simple_channels", "complex_channels", "k_exc", "noise_mode",
             "noise_scale", "noise_level", "ksize", "image_size", "visual_degrees",
             "sf_corr", "sf_max", "sf_min", "rand_param", "gabor_seed")
DEFAULT_ARCH = dict(stride=4, simple_channels=256, complex_channels=256, k_exc=25,
                    noise_mode="neuronal", noise_scale=0.35, noise_level=0.07, ksize=25,
                    image_size=224, visual_degrees=8)


class VOneClassifier(nn.Module):
    """Вход: ImageNet-нормализованный тензор (как у TileResNet/STL-лоадеров).
    Внутри переводим в препроцессинг VOneNet: (x01 - 0.5) / 0.5 (так обучался официальный чекпойнт)."""

    def __init__(self, num_classes=10, **arch):
        super().__init__()
        self.arch = {**DEFAULT_ARCH, **{k: v for k, v in arch.items() if k in ARCH_KEYS}}
        self.net = VOneNet(**self.arch)
        self.net.model.fc = nn.Linear(self.net.model.fc.in_features, num_classes)
        self.register_buffer("mean", IMNET_MEAN.clone())
        self.register_buffer("std", IMNET_STD.clone())

    def set_noise(self, on: bool):
        vb = self.net.vone_block
        vb.set_noise_mode(self.arch["noise_mode"] if on else None,
                          self.arch["noise_scale"], self.arch["noise_level"])

    def forward(self, x, mode="clas"):
        x01 = x * self.std + self.mean
        return self.net((x01 - 0.5) / 0.5)


def from_imagenet_ckpt(path, num_classes=10, image_size=224, visual_degrees=8):
    """Официальный ImageNet-чекпойнт -> VOneClassifier с новой fc на num_classes."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    flags = ck.get("flags", {})
    arch = {k: flags[k] for k in ARCH_KEYS if k in flags}
    arch.update(image_size=image_size, visual_degrees=visual_degrees)
    m = VOneClassifier(num_classes=num_classes, **arch)
    state = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
    state = {k: v for k, v in state.items() if not k.startswith("model.fc.")}   # 1000-классовую fc выкидываем
    missing, unexpected = m.net.load_state_dict(state, strict=False)
    assert all(k.startswith("model.fc.") for k in missing), f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    print(f"[VOne] загружен {path}; arch={m.arch}")
    return m


def save_vone(model: VOneClassifier, path):
    torch.save({"state_dict": model.state_dict(), "arch": model.arch}, path)


def load_vone_classifier(path, device, noise_on=True, num_classes=10, **_):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = VOneClassifier(num_classes=num_classes, **ck.get("arch", {})).to(device)
    m.load_state_dict(ck["state_dict"])
    m.set_noise(noise_on)
    return m.eval()
