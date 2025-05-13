import torch
import torch.nn as nn
import torch.nn.functional as F

# get_stage_channel, bn_before_relu 없음

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        '''
        Channel Shuffle
        1st phase : N, C, H, W 
        2nd phase : N/g, C/g, H, W
        3rd phase : N, C/g, H/g, W
        4th phase : N, C, H, W 
        '''
        N, C, H, W = x.size()
        g = self.groups
        return x.reshape(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
    
class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, is_last=False):
        super(BottleNeck, self).__init__()
        self.is_last = is_last
        self.stride = stride

        mid_planes = int(out_planes / 4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, groups=g, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)

        self.conv2 = nn.Conv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        preact = torch.cat([out, res], 1) if self.stride == 2 else out + res
        out = F.relu(preact)
        if self.is_last:
            return out, preact
        else:
            return out

class ShuffleNet(nn.Module):
    def __init__(self, out_planes, num_blocks, groups, num_classes=100):
        super(ShuffleNet, self).__init__()
        self.out_planes = out_planes
        self.num_blocks = num_blocks
        self.groups = groups

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24

        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)
        self.stage_channels = out_planes
        
    def get_stage_channels(self):
        return [24] + list(self.stage_channels[:])
    
    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(
                BottleNeck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups, is_last=(i == num_blocks - 1),)
            )
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out = F.avg_pool2d(out, 4)
        out = out.reshape(out.size(0), -1)
        f4 = out
        out = self.linear(out)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
        feats["pooled_feat"] = f4
        
        return out, feats

def ShuffleV1(**kwargs):
    return ShuffleNet(out_planes=[240, 480, 960], num_blocks=[4, 8, 4], groups=3, **kwargs), "shufflenetv1"

if __name__ == "__main__":
    
    x = torch.randn(2, 3, 32, 32)
    net, _ = ShuffleV1(num_classes=100)
    logits, feats = net(x)
    print(feats["feats"][-1].shape)