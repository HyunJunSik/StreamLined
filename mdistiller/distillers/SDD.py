import torch
import torch.nn as nn
import torch.nn.functional as F

from .distiller import Distiller
from .DKD import dkd_loss as dkd_loss_origin

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none')
        * (temperature ** 2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none") 
        * (temperature**2) 
        / target.shape[0]
    )
    

    tckd_loss = torch.sum(tckd_loss, dim=1)
    nckd_loss = torch.sum(nckd_loss, dim=1)
    return alpha * tckd_loss + beta * nckd_loss

def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
    
    # from B x C x N to N * B x C
    # Here, N is the number of decoupled region
    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)
    
    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    
    target_r = target.repeat(out_t_multi.shape[2])
    
    # calculate distillation loss
    loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature)
    
    # find the complementary and consistent local distillation loss
    out_t_predict = torch.argmax(out_t, dim=1)
    
    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r
    
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target
    
    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(out_t_multi.shape[2])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(out_t_multi.shape[2])
    
    # global true local wrong
    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False
    
    gt_lw = mask_false
    
    # global wrong local true
    
    mask_true[global_prediction_true_mask_repeat] = False
    mask_true[0:len(target)] = False
    
    gw_lt = mask_true
    
    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r
    
    index = torch.zeros_like(loss).float()
    
    # global wrong local wrong
    
    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false
    
    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true
    
    assert torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt) == target_r.shape[0]
    
    # modify the weight of complementary terms
    
    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2.0
    index[gt_lw] = 2.0
    
    loss = torch.sum(loss * index)
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1).float().cuda()
    return loss
    
def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class SDD_DKD(Distiller):
    def __init__(self, student, teacher):
        super(SDD_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = 2.0
        self.alpha = 3.0
        self.beta = 6.0
        self.temperature = 4.0
        self.warmup = 20
        
    def forward_train(self, image, target, epoch):
        logits_student, feats, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, feats, patch_t = self.teacher(image)
        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(epoch / self.warmup, 1.0) * multi_dkd(
            patch_s,
            patch_t,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_dkd
        }
        
        return logits_student, logits_teacher, losses_dict