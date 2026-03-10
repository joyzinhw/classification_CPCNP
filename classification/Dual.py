import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        ca = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        x = x * self.sigmoid_channel(ca)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)

        return x * self.sigmoid_spatial(self.conv_spatial(sa))



class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        scales=4,
        use_cbam=False
    ):
        super().__init__()

        width = planes
        self.scales = scales
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, width * scales, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scales)

        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, 3, padding=1, bias=False)
            for _ in range(scales - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(width) for _ in range(scales - 1)
        ])

        self.conv3 = nn.Conv2d(
            width * scales,
            planes * self.expansion,
            1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.gelu = nn.GELU()
        self.downsample = downsample
        self.cbam = CBAM(planes * self.expansion) if use_cbam else None

        if stride > 1:
            self.pool = nn.AvgPool2d(3, stride=stride, padding=1)

    def forward(self, x):
        identity = x

        out = self.gelu(self.bn1(self.conv1(x)))
        splits = torch.chunk(out, self.scales, dim=1)

        ys = []
        for i in range(self.scales):
            y = splits[i]

            if self.stride > 1:
                y = self.pool(y)

            if i > 0:
                y = y + ys[i - 1]
                y = self.gelu(self.bns[i - 1](self.convs[i - 1](y)))

            ys.append(y)

        out = self.bn3(self.conv3(torch.cat(ys, dim=1)))

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.gelu(out + identity)



class Res2Net_LungROI(nn.Module):
    def __init__(self, num_classes=3, width=16, scales=4):
        super().__init__()
        self.inplanes = 32

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.gelu = nn.GELU()

        # Menos blocos no início
        self.layer1 = self._make_layer(
            planes=width, blocks=1, stride=1, scales=scales, use_cbam=False
        )

        self.layer2 = self._make_layer(
            planes=width * 2, blocks=1, stride=2, scales=scales, use_cbam=False
        )

        # CBAM só nos estágios profundos
        self.layer3 = self._make_layer(
            planes=width * 4, blocks=2, stride=2, scales=scales, use_cbam=True
        )

        self.layer4 = self._make_layer(
            planes=width * 8, blocks=2, stride=2, scales=scales, use_cbam=True
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)  # mais regularização
        self.fc = nn.Linear(
            width * 8 * Res2NetBottleneck.expansion,
            num_classes
        )

    def _make_layer(self, planes, blocks, stride, scales, use_cbam):
        downsample = None
        out_channels = planes * Res2NetBottleneck.expansion

        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            Res2NetBottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                scales=scales,
                use_cbam=use_cbam
            )
        ]

        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(
                Res2NetBottleneck(
                    self.inplanes,
                    planes,
                    scales=scales,
                    use_cbam=use_cbam
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

