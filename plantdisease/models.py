"""Models: custom CNN with optional SE attention, ResNet18, ViT-B/16, and two-stage freezing (Section 7)."""

from torch import nn
from torchvision import models

from .config import CFG, CLASSES


class SEBlock(nn.Module):
    # Squeeze-and-Excitation channel attention.
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(
            x.mean(dim=(2, 3))
        )  # squeeze: one value per channel -> attention weight in (0, 1)
        return x * weights[:, :, None, None]  # excite: re-weight the channels


class CustomCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3, attention=False):
        super().__init__()
        layers = []
        for c_in, c_out in [
            (3, 32),
            (32, 64),
            (64, 128),
            (128, 256),
        ]:  # 224 -> 112 -> 56 -> 28 -> 14
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            if attention:
                layers.append(SEBlock(c_out))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))  # global average pooling
        return self.fc(self.dropout(x))


def build_model(cfg, pretrained=True):
    if cfg["arch"] in ("cnn", "se_cnn"):
        return CustomCNN(
            len(CLASSES), cfg["dropout"], attention=cfg["arch"] == "se_cnn"
        )
    if cfg["arch"] == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    if cfg["arch"] == "vit_b_16":
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASSES))
        return model
    raise ValueError(cfg["arch"])


HEAD_PREFIX = {"resnet18": ("fc.",), "vit_b_16": ("heads.",)}
LAYER_OF = {  # last convolutional feature maps of every architecture (for Grad-CAM)
    "cnn": lambda m: m.features,
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,
}


def set_trainable(model, arch, stage):
    # stage 'head': only the new classifier is trained; stage 'finetune': (part of) the backbone is unfrozen.
    if arch not in HEAD_PREFIX:  # models trained from scratch: everything is trainable
        return
    n = CFG["vit_trainable_blocks"]
    vit_blocks = (
        *tuple(f"encoder.layers.encoder_layer_{i}." for i in range(12 - n, 12)),
        "encoder.ln.",
    )
    for name, p in model.named_parameters():
        if stage == "head":
            p.requires_grad = name.startswith(HEAD_PREFIX[arch])
        elif arch == "resnet18":
            p.requires_grad = True
        else:
            p.requires_grad = name.startswith(HEAD_PREFIX[arch] + vit_blocks)


def count_params(model, trainable_only=False):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
