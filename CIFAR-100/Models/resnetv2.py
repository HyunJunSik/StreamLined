import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
                
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.inplanes, planes, stride, i == num_blocks - 1))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        return [bn2, bn3, bn4]
    
    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        f4 = out
        out = self.avgpool(out)
        avg = out.reshape(out.size(0), -1)
        out = self.fc(avg)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3, f4]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre, f4_pre]
        feats["pooled_feat"] = avg

        return out, feats
    

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model, "resnet50"


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model, "resnet101"

def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model, "resnet152"

if __name__ == "__main__":
    model, _ = resnet50(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    logits, feats = model(x)
    print(feats["feats"][-1].shape)