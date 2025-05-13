import torch
import torch.nn as nn
import torch.nn.functional as F

from .distiller import Distiller

def dkd_loss(logits_student, logits_teacher ,target, alpha, beta, temperature):
    
    logits_student = torch.clamp(logits_student, min=-10, max=10)
    logits_teacher = torch.clamp(logits_teacher, min=-10, max=10)
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student)

    # Target Class KD
    # tckd는 student 모델의 정답/비정답 클래스에 대한 예측값과 Teacher의 예측값 KL 계산
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum") 
        * (temperature**2) 
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )   
    
    # Non-Target Class Kd
    # 타겟 클래스의 logit값 감소, 비정답 클래스에 대해서만 소프트맥스 함수 적용
    # 이를 통해, 비정답 클래스에 대한 지식 증류를 강화
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum") 
        * (temperature**2) 
        / target.shape[0]
    )

    
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    '''
    주어진 logits 텐서와 타겟 레이블 통해, 타겟 레이블에 해당하는 위치에 True, 나머지 위치는 False 값을 가지는 마스크 생성
    '''
    # target.reshape(-1)를 통해 타겟 텐서를 1차원으로 변환. 이는 scatter 함수를 사용하기 위함임
    target = target.reshape(-1)
    # torch.zeros_like : logits와 동일한 크기를 가지는 0 원소만 가지는 새로운 텐서 생성
    # target.unsqueeze(1) : 타겟 텐서의 각 원소에 대해 새로운 차원을 추가하여 2차원 텐서로 만듦
    # scatter_() : 첫 번째 차원(dim=1)을 따라서, target 텐서에서 지정한 인덱스 위치에 1을 채우고, 나머지 위치에는 0을 남김.
    #              이 과정에서 각 레이블에 해당하는 위치만 1로 표시되고, 나머지는 0으로 표시된 마스크 생성
    '''
    tensor([[-1.2573, -0.5291, -0.5464,  0.2153, -0.4179],
        [ 0.9327, -0.0868, -0.0592,  1.5823,  0.1022],
        [-0.0934, -2.1616, -0.6627,  1.4291, -1.2545]])
    가 logits일때, 
    tensor([2, 1, 4]) 가 label 이라면 이를 2차원으로 바꿔서 [[2], [1], [4]]로 unsqueeze.
    
    이후, scatter까지 진행 및 bool()로 변환하면
    tensor([[False, False,  True, False, False],
        [False,  True, False, False, False],
        [False, False, False, False,  True]]) 로 변환
    '''
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    '''
    정답 클래스와 비정답 클래스에 대한 예측값을 분리하여 합산하고 이를 이어붙임.
    이 과정에서 정답 및 비정답에 대한 정보를 분리함
    '''
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(Distiller):
    '''
    Decoupled Knowledge Distillation(CVPR 2022)
    '''
    def __init__(self, student, teacher):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = 2.0
        self.alpha = 3.0
        self.beta = 6.0
        self.temperature = 4.0
        self.warmup = 20
    
    def forward_train(self, image, target, epoch):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_dkd,
        }
        
        return logits_student, logits_teacher, losses_dict