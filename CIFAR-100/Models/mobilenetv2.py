import torch
import torch.nn as nn
import math

BN = None

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None
        
        self.stride = stride
        assert stride in [1, 2]
        
        self.use_res_connect = self.stride == 1 and inp == oup
        
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw - linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    def forward(self, x):
        out = x
        for i in range(7):
            out = self.conv[i](out)
        preact = out.clone()
        
        out = self.conv[7](out)
        if self.use_res_connect:
            return x + out, preact
        else:
            return out, preact

class MobileNetV2(nn.Module):
    def __init__(self, T, feature_dim, input_size=32, width_mult=1.0, remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg
        
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]
        
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)
        
        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n-1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))
        
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
        
        # building classifier
        
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.last_channel, feature_dim),
        )
        
        H = input_size // (32  // 2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
        
        self._initialize_weights()
        self.stage_channels = [32, 24, 32, 96, 320]
        # [32, 24, 32, 96, 320]
        self.stage_channels = [int(c * width_mult) for c in self.stage_channels]
    
    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m 
    
    def get_stage_channels(self):
        return self.stage_channels
    
    def forward(self, x):
        out = self.conv1(x)
        f0 = out

        # block 0: Sequential of 1 InvertedResidual (ok to run directly)
        for block in self.blocks[0]:
            out, _ = block(out)

        # block 1
        for block in self.blocks[1]:
            out, pre1 = block(out)
        f1 = out

        # block 2
        for block in self.blocks[2]:
            out, pre2 = block(out)
        f2 = out

        # block 3 (no preact 저장)
        for block in self.blocks[3]:
            out, _ = block(out)

        # block 4
        for block in self.blocks[4]:
            out, pre3 = block(out)
        f3 = out

        # block 5
        for block in self.blocks[5]:
            out, _ = block(out)

        # block 6
        for block in self.blocks[6]:
            out, pre4 = block(out)
        f4 = out

        out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        avg = out
        out = self.classifier(out)

        feats = {
            "feats": [f0, f1, f2, f3, f4],
            "preact_feats": [f0, pre1, pre2, pre3, pre4],
            "pooled_feat": avg,
        }
        return out, feats
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
def mobilenetv2(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model

def mobile_half(num_classes):
    return mobilenetv2(6, 0.5, num_classes), "mobilenetv2"


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    net, _ = mobile_half(num_classes=100)
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, stu_feats = net(dummy)

    # 실제 preact_feats 채널 수 추출
    stu_channels = [f.shape[1] for f in stu_feats["preact_feats"][1:]]
    print(stu_channels)
