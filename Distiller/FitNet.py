import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .distiller import Distiller
from ._common import ConvReg, get_feat_shapes

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd

def SKD_loss_cholesky(logits_student, logits_teacher, tik_factor):
    _, C = logits_student.size()
    # Tikhonov Regularization 1e-4, 1e-5, 1e-6
    lambda_val = tik_factor
    # Normalize logits
    logits_student = F.normalize(logits_student, p=2, dim=1)
    logits_teacher = F.normalize(logits_teacher, p=2, dim=1)

    # Compute Gram Matrix
    student_matrix = torch.mm(logits_student, logits_student.T) / (C)
    teacher_matrix = torch.mm(logits_teacher, logits_teacher.T) / (C)
    
    diff = student_matrix - teacher_matrix
    diff = diff.view(diff.size(0), -1)  # Flatten along feature dimensions

    # Compute covariance matrix with Tikhonov regularization
    cov_matrix = torch.cov(diff.T) + lambda_val * torch.eye(diff.size(1), device=diff.device)
    
    L = torch.linalg.cholesky(cov_matrix)  # O(d^2)
    cov_matrix_inv = torch.cholesky_inverse(L)  # O(d^2)
    
    # Compute Mahalanobis distance loss
    mahalanobis_dist = torch.einsum('bi,ij,bj->b', diff, cov_matrix_inv, diff)
    return torch.sqrt(mahalanobis_dist).mean()  # Mean over batch

class FitNet(Distiller):
    '''
    FitNets :Hints for Thin Deep Nets
    '''
    def __init__(self, student, teacher):
        super(FitNet, self).__init__(student, teacher)
        self.ce_loss_weight = 1.0
        self.feat_loss_weight = 100.0
        self.hint_layer = 2
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, (32, 32)
        )
        
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        
    def get_learnable_parameter(self):
        return super().get_learnable_parameter() + list(self.conv_reg.parameters())
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, epoch):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_feat
        }

        return logits_student, logits_teacher, losses_dict         
        
class FitNet_SKD(Distiller):
    '''
    FitNets :Hints for Thin Deep Nets
    '''
    def __init__(self, student, teacher):
        super(FitNet_SKD, self).__init__(student, teacher)
        self.ce_loss_weight = 0.5
        self.kd_loss_weight = 0.5
        self.tik = 1e-1
        self.temperature = 4
        self.feat_loss_weight = 100.0
        self.hint_layer = 2
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, (32, 32)
        )
        
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
        
    def get_learnable_parameter(self):
        return super().get_learnable_parameter() + list(self.conv_reg.parameters())
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, epoch):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        
        
        pred_teacher = F.softmax(logits_teacher.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()
        
        class_confidence = torch.sum(pred_teacher, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()
         
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * ((kd_loss(
            logits_student, 
            logits_teacher, 
            self.temperature
            ) * mask).mean())

        loss_bc = self.kd_loss_weight * ((SKD_loss_cholesky(
            logits_student, 
            logits_teacher, 
            self.tik
            ) * class_conf_mask).mean())
        
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        loss_feat = self.feat_loss_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_bc" : loss_bc,
            "loss_kl" : loss_kd,
            "loss_kd" : loss_feat
        }

        return logits_student, logits_teacher, losses_dict   