import math
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


# =========================================================
# UTILS
# =========================================================
def _make_divisible(ch, divisor=8):
    return max(divisor, int(ch + divisor / 2) // divisor * divisor)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =========================================================
# BLOCKS
# =========================================================
class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        norm_layer=None,
        activation_layer=None
    ):
        padding = (kernel_size - 1) // 2
        norm_layer = norm_layer or nn.BatchNorm2d
        activation_layer = activation_layer or nn.SiLU

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer()
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, expand_c, squeeze_factor=4):
        super().__init__()
        squeeze_c = max(1, input_c // squeeze_factor)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# =========================================================
# INVERTED RESIDUAL
# =========================================================
class InvertedResidualConfig:
    def __init__(
        self,
        kernel,
        input_c,
        out_c,
        expand_ratio,
        stride,
        use_se,
        drop_rate,
        index,
        width_coefficient
    ):
        self.input_c = _make_divisible(input_c * width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expand_ratio
        self.out_c = _make_divisible(out_c * width_coefficient)
        self.stride = stride
        self.use_se = use_se
        self.drop_rate = drop_rate
        self.index = index


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer):
        super().__init__()
        self.use_res = cnf.stride == 1 and cnf.input_c == cnf.out_c

        layers = OrderedDict()

        if cnf.expanded_c != cnf.input_c:
            layers["expand"] = ConvBNActivation(
                cnf.input_c, cnf.expanded_c, kernel_size=1,
                norm_layer=norm_layer
            )

        layers["dw"] = ConvBNActivation(
            cnf.expanded_c, cnf.expanded_c,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=cnf.expanded_c,
            norm_layer=norm_layer
        )

        if cnf.use_se:
            layers["se"] = SqueezeExcitation(cnf.input_c, cnf.expanded_c)

        layers["project"] = ConvBNActivation(
            cnf.expanded_c, cnf.out_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
        )

        self.block = nn.Sequential(layers)
        self.drop = DropPath(cnf.drop_rate) if self.use_res else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.drop(out)
        return x + out if self.use_res else out


# =========================================================
# EFFICIENTNET 64x64
# =========================================================
class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes=3,
        width_coefficient=0.75,
        depth_coefficient=0.75,
        dropout_rate=0.4,
        drop_connect_rate=0.2,
        in_channels=1
    ):
        super().__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        default_cnf = [
            [3, 16, 16, 1, 1, True, 1],
            [3, 16, 24, 4, 2, True, 1],
            [5, 24, 40, 4, 2, True, 1],
            [3, 40, 64, 4, 1, True, 1],
        ]

        def round_repeats(r):
            return int(math.ceil(r * depth_coefficient))

        layers = OrderedDict()

        # Stem
        layers["stem"] = ConvBNActivation(
            in_channels,
            _make_divisible(16 * width_coefficient),
            stride=1,
            norm_layer=norm_layer
        )

        b = 0
        total_blocks = sum(round_repeats(c[-1]) for c in default_cnf)

        in_c = 16
        for stage, cfg in enumerate(default_cnf):
            for i in range(round_repeats(cfg[-1])):
                stride = cfg[4] if i == 0 else 1
                cnf = InvertedResidualConfig(
                    kernel=cfg[0],
                    input_c=in_c,
                    out_c=cfg[2],
                    expand_ratio=cfg[3],
                    stride=stride,
                    use_se=cfg[5],
                    drop_rate=drop_connect_rate * b / total_blocks,
                    index=f"{stage}{i}",
                    width_coefficient=width_coefficient
                )
                layers[cnf.index] = InvertedResidual(cnf, norm_layer)
                in_c = cfg[2]
                b += 1

        # Top
        last_c = _make_divisible(256 * width_coefficient)
        layers["top"] = ConvBNActivation(
            _make_divisible(in_c * width_coefficient),
            last_c,
            kernel_size=1,
            norm_layer=norm_layer
        )

        self.features = nn.Sequential(layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_c, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =========================================================
# FACTORY
# =========================================================
def efficientnet_nodule64(num_classes=3):
    return EfficientNet(num_classes=num_classes)