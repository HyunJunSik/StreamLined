import torch
import torch.nn as nn
import torch.nn.functional as F
import math

cfg = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [
        [64, 64],
        [128, 128],
        [256, 256, 256, 256],
        [512, 512, 512, 512],
        [512, 512, 512, 512],
    ],
    "S": [[64], [128], [256], [512], [512]],
}

class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], 3)
        self.block1 = self._make_layers(cfg[1], cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], cfg[3][-1])
        
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.stage_channels = [c[-1] for c in cfg]
    
    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return self.stage_channels
    
    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        x = self.pool4(x)
        x = x.reshape(x.size(0), -1)
        f5 = x
        x = self.classifier(x)
        
        feats = {}
        feats["feats"] = [f0, f1, f2, f3, f4]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre, f4_pre]
        feats["pooled_feat"] = f5
        
        return x, feats
    
    @staticmethod
    def _make_layers(cfg, in_channels=3):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, 
                           nn.BatchNorm2d(v), 
                           nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def vgg8(**kwargs):
    return VGG(cfg["S"], **kwargs), "vgg8"

def vgg11(**kwargs):
    return VGG(cfg["A"], **kwargs), "vgg11"

def vgg13(**kwargs):
    return VGG(cfg["B"], **kwargs), "vgg13"

def vgg16(**kwargs):
    return VGG(cfg["D"], **kwargs), "vgg16"

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    net, feats = vgg8(num_classes=100)

    logits, feats = net(x)
    print(feats["preact_feats"][0].shape)
    print(feats["preact_feats"][1].shape)
    print(feats["preact_feats"][2].shape)
    print(feats["preact_feats"][3].shape)
    print(feats["preact_feats"][4].shape)