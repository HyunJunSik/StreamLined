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
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logging(teacher_name, student_name, distiller_name, temp):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 기존 로깅 핸들러 제거 (새로운 로그 파일을 생성하기 위해)
    logging.shutdown()  # 기존 로그 종료
    logging.getLogger().handlers.clear()  # 핸들러 제거

    log_filename = f"./distiller_train_log/experiment_cifar-100_{now}_{teacher_name}_{student_name}_{distiller_name}.log"
    
    logging.basicConfig(
        filename=log_filename, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s'
    )
    logging.info("===== New Logging Session Started =====")

    

def accuracy(output, target, topk=(1,)):
    '''
    topk = 1이라면 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한지 계산 
    topk = (1, 5)라면, 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한 경우를 계산하여
    top1 정확도 구하고, 그 다음으로 높은 5개의 예측 확률을 가진 레이블 중 실제 레이블이 포함되는지 확인하여 top5 정확도 구함
    
    더욱 모델의 성능을 상세하게 평가하기 위한 방법으로, 모델의 성능을 다각도로 이해하고 평가하는 데 도움됨
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
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
        optimizer = optim.SGD(
            # get_learnable_parameter가 distiller에 잘 되어있는지 체크
            self.distiller.module.get_learnable_parameter(),
            lr=0.1,
            momentum=0.9,
            weight_decay=4e-5,
        )
        return optimizer
    
    def train(self):
        epoch_length = 250
        for epoch in range(epoch_length):
            print(f"Epoch: {epoch + 1} / {epoch_length}")
            logging.info(f"Epoch: {epoch + 1} / {epoch_length}")
            total_loss = 0
            total_acc1 = 0
            total_acc5 = 0
            best_acc = 0
            
            if epoch in [150, 180, 210]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    
            dataset_len = len(self.train_loader.dataset)
            self.distiller.train()
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{epoch_length}", leave=False) as pbar:
                for idx, (inputs, labels) in pbar: 
                # for idx, (inputs, labels, index, sample_idx) in pbar: 
                    inputs, labels = inputs.to(device), labels.to(device)
                    # index = index.to(device)
                    # sample_idx = sample_idx.to(device)
                    batch_size = inputs.size(0)
                    self.optimizer.zero_grad()
                    
                    # preds, teacher, losses_dict = self.distiller(inputs, labels, epoch, index, sample_idx)
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
            
            if test_acc1 > best_acc:
                best_acc = test_acc1
                # distiller는 DataPrallel에 의해 모델이 감싸져서 모델의 속성들을 직접적으로 노출안됨.
                # 따라서, module 속성을 통해 원래 모델 접근 가능
                best_model_wts = copy.deepcopy(self.distiller.module.student.state_dict())
            self.distiller.module.student.load_state_dict(best_model_wts)
        return best_model_wts
a = "teacher channel"            
if __package__ is None:
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Distiller import distiller
    from Distiller.KD import KD
    from Distiller.DKD import DKD
    from Distiller.CLKD import CLKD
    from Distiller.MLKD import MLKD_align_3
    from Distiller.SKD import SKD
    from Distiller.FitNet import FitNet, FitNet_SKD
    from Distiller.RKD import RKD
    from Distiller.ReviewKD import ReviewKD
    from Distiller.OFD import OFD
    from Distiller.CRD import CRD
    from Models import vgg, wrn, shufflenet_v1, shufflenet_v2, resnet
    from train import load_dataset_multiapply, load_dataset, load_dataset_crd
    
def main(selected_student, selected_teacher, selected_distiller, batch_size=64, temp=4): 
    
    model_student, student_name = [
        resnet.resnet32(num_classes=100),
        resnet.resnet8x4(num_classes=100),
        resnet.resnet20(num_classes=100),
        shufflenet_v1.ShuffleV1(num_classes=100),
        shufflenet_v2.ShuffleV2(num_classes=100),
        vgg.vgg8(num_classes=100),
        wrn.wrn_40_1(num_classes=100),
        wrn.wrn_16_2(num_classes=100),
        ][selected_student]

    model_teacher, teacher_name = [
        resnet.resnet32x4(num_classes=100), 
        resnet.resnet56(num_classes=100),
        resnet.resnet110(num_classes=100),
        vgg.vgg13(num_classes=100),
        wrn.wrn_40_2(num_classes=100),
        ][selected_teacher]
    
    model_teacher.load_state_dict(load_teacher_param(teacher_name))
    temperature = temp
    train_loader, val_loader = load_dataset()
    # train_loader, val_loader = load_dataset_crd(bz=batch_size)
    
    # CRD model feats
    if selected_student in [0, 2, 6]:
        stu_feats = 64
    elif selected_student in [1]:
        stu_feats = 256
    elif selected_student in [5]:
        stu_feats = 512
    elif selected_student in [7]:
        stu_feats = 128
    elif selected_student in [3]:
        stu_feats = 960
    elif selected_student in [4]:
        stu_feats = 1024
    else:
        stu_feats = 1024
        
    if selected_teacher in [1, 2]:
        tea_feats = 64
    elif selected_teacher in [0]:
        tea_feats = 256
    elif selected_teacher in [4]:
        tea_feats = 512
    elif selected_teacher in [5]:
        tea_feats = 128
    else:
        tea_feats = 2048
    
    distiller = [
        SKD(model_student, model_teacher),
        KD(model_student, model_teacher),
        DKD(model_student, model_teacher),
        CLKD(model_student, model_teacher),
        MLKD_align_3(model_student, model_teacher),
        RKD(model_student, model_teacher),
        ReviewKD(model_student, model_teacher, student_name, teacher_name),
        OFD(model_student, model_teacher),
        CRD(model_student, model_teacher, len(train_loader.dataset), stu_feats, tea_feats),
        ][selected_distiller]
    distiller_name = [
        "SKD",
        "KD",
        "DKD",
        "CLKD",
        "MLKD",
        "RKD",
        "ReviewKD",
        "OFD",
        "CRD",
        ][selected_distiller]
    distiller = torch.nn.DataParallel(distiller.cuda())
    
    print(f"Teacher Model : {teacher_name}, Student Model : {student_name}, Distiller : {distiller_name}")
    setup_logging(teacher_name, student_name, distiller_name, temp=temperature)
    logging.info(f"Running main with student={student_name}, teacher={teacher_name}, distiller={distiller_name}")
    start_time = time.time()
    
    trainer = AugTrainer(distiller, train_loader, val_loader)
    best_model_wts = trainer.train()
    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

    torch.save(best_model_wts, f"model_distillation_pth/best_model_weights_{student_name}_{distiller_name}.pth")
    


def load_teacher_param(model_name):
    model_state_dict_path = f"model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path)
    return model_state_dict

if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    print(f"device : {device}")
    
    parser = argparse.ArgumentParser(description="Run KD")
    parser.add_argument("--selected_student", type=int, default=0, help="Index of the student model")
    parser.add_argument("--selected_teacher", type=int, default=0, help="Index of the teacher model")
    parser.add_argument("--selected_distiller", type=int, default=0, help="Index of the distillation method")
    args = parser.parse_args()
    
    main(
        selected_student=args.selected_student,
        selected_teacher=args.selected_teacher,
        selected_distiller=args.selected_distiller
    )