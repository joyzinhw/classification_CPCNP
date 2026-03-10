import torch
import torch.nn as nn



class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        avg = self.mlp(self.avg_pool(x))
        max = self.mlp(self.max_pool(x))

        channel_attention = self.sigmoid(avg + max)

        x = x * channel_attention

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial = torch.cat([avg_out, max_out], dim=1)

        spatial_attention = self.spatial(spatial)

        x = x * spatial_attention

        return x


# =========================
# BLOCO CONV
# =========================
class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


# =========================
# MODELO FINAL
# =========================
class LungNoduleNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)

        self.attention = CBAM(64)

        self.block3 = ConvBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.block1(x)  # 64 → 32
        x = self.block2(x)  # 32 → 16

        x = self.attention(x)

        x = self.block3(x)  # 16 → 8

        x = self.global_pool(x)

        x = self.classifier(x)

        return x