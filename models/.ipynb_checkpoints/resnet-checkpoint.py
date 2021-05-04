# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet18_10','ResNet18_100']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ops, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ops.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = ops.Conv2d(out_channel, out_channel, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = ops.ReLU()

        self.shortcut = nn.Sequential() # dimension expansion
        if stride != 1 or in_channel != self.expansion*out_channel:
            self.shortcut = nn.Sequential(
                ops.Conv2d(in_channel, self.expansion*out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channel)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out



# +
# resnet18 - no BottleNeck
# -

class ResNet(nn.Module):
    def __init__(self, ops, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = ops.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False) # 첫번째 layer는 no quantization
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ops.ReLU()
        self.layer1 = self._make_layer(ops, block, 64, num_blocks[0], stride=1) # num_blocks = 2
        self.layer2 = self._make_layer(ops, block, 128, num_blocks[1], stride=2) # num_blocks = 2
        self.layer3 = self._make_layer(ops, block, 256, num_blocks[2], stride=2) # num_blocks = 2
        self.layer4 = self._make_layer(ops, block, 512, num_blocks[3], stride=2) # num_blocks = 2
        self.linear = ops.Linear(512*block.expansion, num_classes) # 마지막 layer도 quantization

    def _make_layer(self, ops, block, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(ops, self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers) # q_sym, h-swish 없을때는 뭘 써도 상관 x

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_10(ops):
    model= ResNet(ops, BasicBlock, [2,2,2,2])
    return model

def ResNet18_100(ops):
    model = ResNet(ops, BasicBlock, [2,2,2,2], num_classes=100)
    return model

def test():
    net = ResNet18_100()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()

# num_blocks = [3,4,6,3]
# stride = [1,2,2,2]
# for i in range(4):
#     strides = [stride[i]] + [1]*(num_blocks[i]-1)
#     print(strides)


