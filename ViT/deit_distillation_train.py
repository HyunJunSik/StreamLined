'''
1. 기본 ViT를 가지고, Soft Distill, Hard Distill 수행
2. DeiT를 가지고 Hard Distill 수행
'''
import os
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import logging
import copy
from tqdm import tqdm
from datetime import datetime
import time
import sys
from os import path
import argparse
import timm
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.optim.lr_scheduler import LambdaLR
import math

device = torch.device("hpu")
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from imagenet_train import get_imagenet_dataloaders_strong
    from Foundation.ViT import ViT_T_patch16_224, DeiT_T_patch16_224
    from Distiller.Vit_distill import VIT, one_VIT

def warmup_cosine_annealing_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # 워밍업 단계: 선형적으로 학습률 증가
            return float(current_epoch) / float(max(1, warmup_epochs))
        # 코사인 에널링 단계: 학습률 감소
        return 0.5 * (1.0 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return LambdaLR(optimizer, lr_lambda)

def setup_logging(teacher_name, student_name, distiller_name):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 기존 로깅 핸들러 제거 (새로운 로그 파일을 생성하기 위해)
    logging.shutdown()  # 기존 로그 종료
    logging.getLogger().handlers.clear()  # 핸들러 제거

    log_filename = f"./distiller_train_log/experiment_imagenet_{now}_{teacher_name}_{student_name}_{distiller_name}.log"
    
    logging.basicConfig(
        filename=log_filename, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s'
    )
    logging.info("===== New Logging Session Started =====")

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(100.0 / batch_size))
        return res

def validate(val_loader, distiller):
    criterion = nn.CrossEntropyLoss()
    dataset_len = len(val_loader.dataset)
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0 
    
    distiller.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = distiller(inputs, labels)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc1 += acc1.item() * batch_size
            total_acc5 += acc5.item() * batch_size
    val_loss = total_loss / dataset_len
    val_acc1 = total_acc1 / dataset_len
    val_acc5 = total_acc5 / dataset_len
    return val_loss, val_acc1, val_acc5

class AugTrainer(object):
    def __init__(self, distiller, train_loader, val_loader):
        self.train_loader = train_loader
        self.distiller = distiller
        self.val_loader = val_loader
        self.best_acc = -1
        self.optimizer = self.init_optimizer()
        
    def init_optimizer(self):
        optimizer = optim.AdamW(self.distiller.get_learnable_parameter(), 
                                lr=5e-4, 
                                eps=1e-8,
                                betas=(0.9, 0.999), 
                                weight_decay=0.05,
                                )
        return optimizer

    def train(self):
        epoch_length = 300
        warmup_epochs = 10
        scheduler = warmup_cosine_annealing_scheduler(self.optimizer, warmup_epochs, epoch_length)
        
        for epoch in range(epoch_length):
            print(f"Epoch: {epoch + 1} / {epoch_length}")
            logging.info(f"Epoch: {epoch + 1} / {epoch_length}")
            total_loss = 0
            total_acc1 = 0
            total_acc5 = 0
            best_acc = 0
            
            if epoch % 30 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1    
            
            dataset_len = len(self.train_loader.dataset)
            self.distiller.train()
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{epoch_length}", leave=False) as pbar:
                for idx, (inputs, labels) in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size = inputs.size(0)
                    self.optimizer.zero_grad()

                    preds, teacher, losses_dict = self.distiller(inputs, labels, epoch)
                    loss = sum([l.mean() for l in losses_dict.values()])
                    loss.backward()
                    self.optimizer.step()
                   
                    acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
                    total_loss += loss.item() * batch_size
                    total_acc1 += acc1.item() * batch_size
                    total_acc5 += acc5.item() * batch_size
                
            train_loss = total_loss / dataset_len
            train_acc1 = total_acc1 / dataset_len
            train_acc5 = total_acc5 / dataset_len
            
            print(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
            logging.info(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
            # validate
            test_loss, test_acc1, test_acc5 = validate(self.val_loader, self.distiller)
            print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
            logging.info(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
            
            scheduler.step()
            
            if test_acc1 > best_acc:
                best_acc = test_acc1
                best_model_wts = copy.deepcopy(self.distiller.student.state_dict())
            self.distiller.student.load_state_dict(best_model_wts)
        return best_model_wts

def main(model_type, distillation_type):
    train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(1024, 128, 32)
    
    model_teacher = timm.create_model('regnety_160.tv2_in1k', pretrained=True)
    data_config = timm.data.resolve_model_data_config(model_teacher)
    transform = timm.data.create_transform(**data_config, is_training=False)
    teacher_name = "Regnety"
    
    if model_type == 'vit':
        model_student, student_name = ViT_T_patch16_224()
        distiller = one_VIT(model_student, model_teacher, model_type, distillation_type)
    elif model_type == 'deit':
        model_student, student_name = DeiT_T_patch16_224()
        distiller = VIT(model_student, model_teacher, model_type, distillation_type)

    if model_type == 'vit':
        if distillation_type == 'soft':
            distiller_name = "soft_distill"
        elif distillation_type == 'hard':
            distiller_name = "hard_distill"
        elif distillation_type == 'SKD':
            distiller_name = "SKD"
    elif model_type == 'deit':
        distiller_name = "deit"
    distiller = distiller.to(device)
    
    print(f"Teacher Model : {teacher_name}, Student Model : {student_name}, Distiller : {distiller_name}")
    setup_logging(teacher_name, student_name, distiller_name)
    logging.info(f"Running main with student={student_name}, teacher={teacher_name}, distiller={distiller_name}")
    start_time = time.time()
    trainer = AugTrainer(distiller, train_loader, val_loader)
    best_model_wts = trainer.train()

    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

    torch.save(best_model_wts, f"model_distillation_pth/best_model_weights_{student_name}_{distiller_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Distillation Trainer")
    parser.add_argument("--model_type", type=str, required=True, choices=["vit", "deit"])
    parser.add_argument("--distillation_type", type=str, choices=["soft", "hard", "SKD", "none"], default="none")
    args = parser.parse_args()
    
    distill_type = None if args.distillation_type == "none" else args.distillation_type

    main(model_type=args.model_type, distillation_type=distill_type)