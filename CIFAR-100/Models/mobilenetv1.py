import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV1(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNetV1, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(512, 100)
    
    def forward(self, x, is_feat=False):
        # Stage 1: 처음 4개 블록 (0~3)
        feat1 = self.model[3][:-1](self.model[0:3](x))  # conv_bn + dw x 3 (128 채널)

        # Stage 2: 4~5번 블록 (256 채널)
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))

        # Stage 3: 6~11번 블록 (512 채널)
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))

        # Stage 4: Adaptive Pool
        feat4 = self.model[12](F.relu(feat3))

        avg = feat4.view(feat4.size(0), -1)
        out = self.fc(avg)

        feats = {
            "pooled_feat": avg,
            "feats": [F.relu(feat1), F.relu(feat2), F.relu(feat3), F.relu(feat4)],
            "preact_feats": [feat1, feat2, feat3, feat4]
        }
        return out, feats
    
    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]   # feat1 직전 BN
        bn2 = self.model[5][-2]   # feat2 직전 BN
        bn3 = self.model[11][-2]  # feat3 직전 BN
        bn4 = None                # Adaptive Pool은 BN 없음
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 512]

def mobilenetv1(**kwargs):
    model = MobileNetV1()
    return model, "mobilenetv1"

if __name__ == "__main__":
    model, _ = mobilenetv1()
    x = torch.randn(2, 3, 32, 32)
    logits, feats = model(x)
    print(feats["preact_feats"][0].shape)
    print(feats["preact_feats"][1].shape)
    print(feats["preact_feats"][2].shape)
    print(feats["preact_feats"][3].shape)
    