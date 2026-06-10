"""
ResNet-18 backbone with a small classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


class ResNetClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def build_model(pretrained: bool = True) -> nn.Module:
    return ResNetClassifier(pretrained=pretrained)