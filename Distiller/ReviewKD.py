import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from .distiller import Distiller

# if 'x4' in model: 
#         student = build_resnetx4_backbone(depth = int(model[6:-2]), num_classes = num_classes)
#         in_channels = [64,128,256,256]
#         out_channels = [64,128,256,256]
#         shapes = [1,8,16,32]
#     elif 'ResNet50' in model:
#         student = ResNet50(num_classes = num_classes)
#         in_channels = [16,32,64,64]
#         out_channels = [16,32,64,64]
#         shapes = [1,8,16,32,32]
#         assert False
#     elif 'resnet' in model:
#         student = build_resnet_backbone(depth = int(model[6:]), num_classes = num_classes)
#         in_channels = [16,32,64,64]
#         out_channels = [16,32,64,64]
#         shapes = [1,8,16,32,32]
#     elif 'vgg' in model:
#         student = build_vgg_backbone(depth = int(model[3:]), num_classes = num_classes)
#         in_channels = [128,256,512,512,512]
#         shapes = [1,4,4,8,16]
#         if 'ResNet50' in teacher:
#             out_channels = [256,512,1024,2048,2048]
#             out_shapes = [1,4,8,16,32]
#         else:
#             out_channels = [128,256,512,512,512]
#     elif 'mobile' in model:
#         student = mobile_half(num_classes = num_classes)
#         in_channels = [12,16,48,160,1280]
#         shapes = [1,2,4,8,16]
#         if 'ResNet50' in teacher:
#             out_channels = [256,512,1024,2048,2048]
#             out_shapes = [1,4,8,16,32]
#         else:
#             out_channels = [128,256,512,512,512]
#             out_shapes = [1,4,4,8,16]
#     elif 'shufflev1' in model:
#         student = ShuffleV1(num_classes = num_classes)
#         in_channels = [240,480,960,960]
#         shapes = [1,4,8,16]
#         if 'wrn' in teacher:
#             out_channels = [32,64,128,128]
#             out_shapes = [1,8,16,32]
#         else:
#             out_channels = [64,128,256,256]
#             out_shapes = [1,8,16,32]
#     elif 'shufflev2' in model:
#         student = ShuffleV2(num_classes = num_classes)
#         in_channels = [116,232,464,1024]
#         shapes = [1,4,8,16]
#         out_channels = [64,128,256,256]
#         out_shapes = [1,8,16,32]
#     elif 'wrn' in model:
#         student = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=num_classes)
#         r=int(model[-1:])
#         in_channels = [16*r,32*r,64*r,64*r]
#         out_channels = [32,64,128,128]
#         shapes = [1,8,16,32]

def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

class ReviewKD(Distiller):
    def __init__(self, student, teacher, student_name, teacher_name):
        super(ReviewKD, self).__init__(student, teacher)
        '''
        Baseline shapes, out_shapes, in_channels, out_channels
        '''
        self.shapes= [1, 8, 16, 32, 32]
        self.out_shapes = [1, 8, 16, 32, 32]
        in_channels = [16, 32, 64, 64]
        out_channels = [16, 32, 64, 64]

        if "mobilenetv1" in student_name:
            in_channels = [128, 256, 512, 512, 512]
            self.shapes = [1,2,4,8,16]
            if "resnet50" in teacher_name:
                out_channels = [256, 512, 1024, 2048, 2048]
                self.out_shapes = [1, 4, 8, 16, 32]
            else:
                out_channels = [128, 256, 512, 512, 512]
                self.out_shapes = [1, 4, 4, 8, 16]
        '''
        Homogeneous
        resnet*x4
        - in_channels : [64, 128, 256, 256]
        - out_channels : [64, 128, 256, 256]
        - shapes = [1, 8, 16, 32]
        - out_shapes = None
        resnet
        - in_channels : [16, 32, 64, 64]
        - out_channels : [16, 32, 64, 64]
        - shapes : [1, 8, 16, 32, 32]
        - out_shapes = None
        vgg
        - in_channels : [128, 256, 512, 512, 512]
        - out_channels : [128, 256, 512, 512, 512]
        - shapes : [1, 4, 4, 8, 16]
        - out_shapes = None
        '''
            

        # if "shufflenetv1" in student_name:
        #     self.shapes = [1, 4, 8, 16]
        #     self.out_shapes = [1, 8, 16, 32]
        #     in_channels = [240, 480, 960, 960]
        #     if "resnet32x4" in teacher_name:
        #         out_channels = [64, 128, 256, 256]
        #     else:
        #         out_channels = [32, 64, 128, 128]  
        # elif "shufflenetv2" in student_name:
        #     self.shapes = [1, 4, 8, 16]
        #     self.out_shapes = [1, 8, 16, 32]
        #     in_channels = [116, 232, 464, 1024]
        #     out_channels = [64, 128, 256, 256]
        # elif "mobilenetv2" in student_name:
        #     self.shapes = [1, 2, 4, 8, 16]
        #     in_channels = [12, 16, 48, 160, 1280]
        #     if "vgg13" in teacher_name:
        #         self.out_shapes = [1, 4, 4, 8, 16]
        #         out_channels = [128, 256, 512, 512, 512]
        #     else:
        #         self.out_shapes = [1, 4, 8, 16, 32]
        #         out_channels = [256, 512, 1024, 2048, 2048]
            
        
        self.ce_loss_weight = 1.0
        self.reviewkd_loss_weight = 1.0
        self.warnup_epochs = 20
        self.stu_preact = False
        self.max_mid_channel = 512

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]
    
    def get_learnable_parameter(self):
        return super().get_learnable_parameter() + list(self.abfs.parameters())
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, epoch):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
        
        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
            ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
            
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(epoch / self.warnup_epochs, 1.0)
            * hcl_loss(results, features_teacher)
        )
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_reviewkd,
        }
        return logits_student, logits_teacher, losses_dict

class ABF(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, fuse):
        super(ABF, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channels * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)
    
    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        
        x = self.conv1(x)

        if self.att_conv is not None:
            # 💡 1. y가 None이 아니고, shape이 x와 다르면 맞춰줌
            if y.shape[2:] != x.shape[2:]:
                y = F.interpolate(y, size=(h, w), mode="nearest")

            # 2. concat & attention fusion
            z = torch.cat([x, y], dim=1)  # x와 y는 같은 spatial shape
            z = self.att_conv(z)

            # 3. attention weighted sum
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)

        # 💡 4. out_shape에 맞춰서 최종 interpolation
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, size=(out_shape, out_shape), mode="nearest")

        y = self.conv2(x)
        return y, x
