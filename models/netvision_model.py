import torch
import torch.nn as nn
import math
from torch import Tensor

# 注意：SELayer 类可以保留在文件中但不使用，或者直接删除。

# ==========================================
# 1. 基础组件：通道重排与 Ghost 模块
# ==========================================
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# ==========================================
# 2. 核心模块：LRBBlock (已移除 SE 模块)
# ==========================================
class LRBBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super(LRBBlock, self).__init__()
        self.stride = stride

        new_out_channel = out_chs // 2

        if stride == 1:
            new_channel = in_chs // 2
            self.branch2 = nn.Sequential(
                GhostModule(new_channel, mid_chs, relu=True),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )
            self.shortcut = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )

            self.branch2 = nn.Sequential(
                GhostModule(in_chs, mid_chs, relu=True),
                nn.Conv2d(mid_chs, mid_chs, 3, stride=stride, padding=1, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(new_out_channel)
            )

        # 删除了 self.se = SELayer(out_chs)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2) + self.shortcut(x2)), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x) + self.shortcut(x)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        # 删除了 return self.se(out)，直接返回拼接后的结果
        return out


# ==========================================
# 3. 主网络：NetVision (结构保持不变)
# ==========================================
class NetVision(nn.Module):
    def __init__(self, num_classes=8):
        super(NetVision, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            LRBBlock(16, 32, 32, stride=2),
            LRBBlock(32, 48, 32, stride=1)
        )

        self.layer2 = nn.Sequential(
            LRBBlock(32, 64, 64, stride=2),
            LRBBlock(64, 96, 64, stride=1)
        )

        self.layer3 = nn.Sequential(
            LRBBlock(64, 128, 128, stride=2),
            LRBBlock(128, 192, 128, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        return x