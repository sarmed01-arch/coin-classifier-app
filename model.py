
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out

class ResidualGroup(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels)
        )

    def forward(self, x):
        return self.blocks(x)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        logits = self.attn_conv(x)
        logits = logits.view(b, 1, -1)
        weights = F.softmax(logits, dim=-1)

        feats = x.view(b, c, -1)
        context = torch.sum(feats * weights, dim=-1)

        return context

class CoinNetColab(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.residual_group = ResidualGroup(512)
        self.attention = SpatialAttention(512)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = self.residual_group(x)
        context = self.attention(x)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CoinNetColab(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class
