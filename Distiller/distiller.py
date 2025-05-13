import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode

        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self
    
    def get_learnable_parameter(self):
        return [v for k, v in self.student.named_parameters()]
    
    def forward_train(self, **kwargs):
        raise NotImplementedError()
    
    def forward_test(self, image):
        output, _ = self.student(image)
        return output

    def forward(self, image, label, epoch=None, index=None, sample_index=None):
        if self.training:
            if index == None:
                return self.forward_train(image, label, epoch)
            else:       
                # for CRD
                return self.forward_train(image, label, epoch, index, sample_index)
        return self.forward_test(image)

class Vanilla(nn.Module):

    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student
    
    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]
    
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, loss
    
    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])
    