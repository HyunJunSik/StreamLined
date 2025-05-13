import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from .distiller import Distiller

def CLKD_loss(logits_student, logits_teacher):
    '''
    instance_distill_loss, class_distill_loss, class_correlation_loss
    '''

    student_norm = F.normalize(logits_student, p=2, dim=1)
    teacher_norm = F.normalize(logits_teacher, p=2, dim=1)
    instance_loss = F.mse_loss(student_norm, teacher_norm)
    
    
    t_stu = torch.t(logits_student)
    t_tea = torch.t(logits_teacher)
    
    t_student_norm = F.normalize(t_stu, p=2, dim=1)
    t_teacher_norm = F.normalize(t_tea, p=2, dim=1)
    class_loss = F.mse_loss(t_student_norm, t_teacher_norm)
    
    N, C = logits_student.shape
    
    student_mean = torch.mean(logits_student, dim=0)
    teacher_mean = torch.mean(logits_teacher, dim=0)
    
    diff_s = logits_student - student_mean.unsqueeze(0)
    B_s = diff_s.T @ diff_s / (C - 1)

    diff_t = logits_teacher - teacher_mean.unsqueeze(0)
    B_t = diff_t.T @ diff_t / (C - 1)

    diff = B_s - B_t
    diff_norm = torch.norm(diff) 
    class_corr_loss = (1 / (C**2)) * diff_norm 


    return instance_loss, class_loss, class_corr_loss

class CLKD(Distiller):
    '''
    
    '''
    def __init__(self, student, teacher):
        super(CLKD, self).__init__(student, teacher)
        self.lamb = 0.1
        self.mu = 0.5
        self.vu = 0.4
        self.beta = 2.0
    
    def forward_train(self, image, target, epoch):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        ins_loss, cla_loss, clcor_loss = CLKD_loss(logits_student, logits_teacher)
        loss_ce = self.lamb * F.cross_entropy(logits_student, target)
        loss_kd = self.mu * (ins_loss + self.beta * cla_loss) + self.vu * clcor_loss
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_kd,
        }

        return logits_student, logits_teacher, losses_dict