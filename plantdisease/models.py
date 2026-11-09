"""Models: custom CNN with optional SE attention, ResNet18, ViT-B/16, and two-stage freezing (Sections 7 and 10)."""

from torch import Tensor, nn
from torchvision import models

from .config import DATA_CONFIG
from .recipe import RunConfig


class SEBlock(nn.Module):
    # Squeeze-and-Excitation channel attention.
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.fc(
            x.mean(dim=(2, 3))
        )  # squeeze: one value per channel -> attention weight in (0, 1)
        return x * weights[:, :, None, None]  # excite: re-weight the channels


class CustomCNN(nn.Module):
    def __init__(
        self, num_classes: int, dropout: float = 0.3, attention: bool = False
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x).mean(dim=(2, 3))  # global average pooling
        return self.fc(self.dropout(x))


def build_model(cfg: RunConfig, pretrained: bool = True) -> nn.Module:
    k = len(DATA_CONFIG["classes"])  # number of outputs
    if cfg.arch in ("cnn", "se_cnn"):
        return CustomCNN(k, cfg.dropout, attention=cfg.arch == "se_cnn")
    if cfg.arch == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, k)
        return model
    if cfg.arch == "vit_b_16":
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.heads.head = nn.Linear(model.heads.head.in_features, k)
        return model
    raise ValueError(cfg.arch)


HEAD_PREFIX = {"resnet18": ("fc.",), "vit_b_16": ("heads.",)}
LAYER_OF = {  # last convolutional feature maps of every architecture (for Grad-CAM)
    "cnn": lambda m: m.features,
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,
}


def is_head(cfg: RunConfig, name: str) -> bool:
    # Does parameter `name` belong to the new classifier head?
    return name.startswith(HEAD_PREFIX.get(cfg.arch, ()))


def set_trainable(model: nn.Module, cfg: RunConfig, stage: str) -> None:
    # Stage 'head': only the new classifier is trained. Stage 'finetune': cfg.finetune decides
    # (fine-tuning strategies of Section 10):
    #   'head'    the classifier only (backbone frozen throughout)
    #   'partial' the last cfg.trainable_blocks ViT encoder blocks, the final LayerNorm and the classifier
    #   'full'    every parameter, including patch embedding, position embedding and class token (default)
    if (
        cfg.arch not in HEAD_PREFIX
    ):  # models trained from scratch: everything is trainable
        return
    scope = "head" if stage == "head" else cfg.finetune
    if scope == "full":
        model.requires_grad_(True)
        return
    trainable = HEAD_PREFIX[cfg.arch]
    if scope == "partial":
        n = cfg.trainable_blocks
        blocks = [f"encoder.layers.encoder_layer_{i}." for i in range(12 - n, 12)]
        trainable = (*trainable, *blocks, "encoder.ln.")
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(trainable)


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
