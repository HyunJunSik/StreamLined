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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Distiller import distiller
    from Distiller.KD import KD
    from Distiller.DKD import DKD
    from Distiller.CLKD import CLKD
    from Distiller.AT import AT
    from Distiller.MLKD import MLKD_align_3
    from Distiller.ReviewKD import ReviewKD
    from Distiller.OFD import OFD
    from Distiller.RKD import RKD
    # from Distiller.FitNet import FitNet
    # from Distiller.RKD import RKD
    # from Distiller.ReviewKD import ReviewKD
    from imagenet_train import get_imagenet_dataloaders_strong

def parse_args():
    parser = argparse.ArgumentParser(description="Run Knowledge Distillation Method")
    parser.add_argument("selected_distiller", type=int, help="Selected distiller 0, 1, 2, etc")
    return parser.parse_args()

def setup_logging(teacher_name, student_name, distiller_name, temp):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 기존 로깅 핸들러 제거 (새로운 로그 파일을 생성하기 위해)
    logging.shutdown()  # 기존 로그 종료
    logging.getLogger().handlers.clear()  # 핸들러 제거

    log_filename = f"./distiller_train_log/experiment_imagenet_{now}_{teacher_name}_{student_name}_{distiller_name}_temp_{temp}.log"
    
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
        optimizer = optim.SGD(
            self.distiller.get_learnable_parameter(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )
        return optimizer
    
    def train(self):
        epoch_length = 100
        for epoch in range(epoch_length):
            print(f"Epoch: {epoch + 1} / {epoch_length}")
            logging.info(f"Epoch: {epoch + 1} / {epoch_length}")
            total_loss = 0
            total_acc1 = 0
            total_acc5 = 0
            best_acc = 0
            
            if epoch in [30, 60, 90, 120]:
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
                    
                    # if (epoch + 1) % 40 == 0:
                    #     with torch.no_grad():
                    #         batch_diff = compute_diff(preds, teacher)
                    #         diff_path = os.path.join(diff_path_dir, f"diff_epoch_{epoch + 1}.pt")
                    #         torch.save(batch_diff, diff_path)
                    
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
                best_model_wts = copy.deepcopy(self.distiller.student.state_dict())
            self.distiller.student.load_state_dict(best_model_wts)
        return best_model_wts
    
def load_teacher_param(model_name):
    model_state_dict_path = f"imagenet_model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path)
    return model_state_dict

def main(selected_student, selected_teacher, selected_distiller, batch_size=64, temp=4):
    
    seed=2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    temperature = temp
    train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(batch_size, 64, 32)
    model_student, student_name = [
        imagenet_resnet.resnet18(pretrained=False,num_classes=1000),
        mobilenetv2.mobilenetv2(),
    ][selected_student]
    model_teacher, teacher_name = [
        imagenet_resnet.resnet34(pretrained=True,num_classes=1000),
        imagenet_resnet.resnet50(pretrained=True,num_classes=1000),
    ][selected_teacher]
    # model_teacher.load_state_dict(load_teacher_param(teacher_name))

    distiller = [
        KD(model_student, model_teacher),
        DKD(model_student, model_teacher),
        CLKD(model_student, model_teacher),
        MLKD_align_3(model_student, model_teacher),
        AT(model_student, model_teacher),
        SKD(model_student, model_teacher, temperature),
        ReviewKD(model_student, model_teacher),
        OFD(model_student, model_teacher),
        RKD(model_student, model_teacher),
        # FitNet(model_student, model_teacher),
        ][selected_distiller]
    
    distiller_name = [
        "KD",
        "DKD",
        "CLKD",
        "MLKD",
        "AT",
        "SKD",
        "ReviewKD",
        "OFD",
        "RKD",
        # "fitnet",
        ][selected_distiller]
    distiller = distiller.to(device)
    
    print(f"Teacher Model : {teacher_name}, Student Model : {student_name}, Distiller : {distiller_name}")
    setup_logging(teacher_name, student_name, distiller_name, temp=temperature)
    logging.info(f"Running main with student={student_name}, teacher={teacher_name}, distiller={distiller_name}")
    start_time = time.time()
    
    trainer = AugTrainer(distiller, train_loader, val_loader)
    best_model_wts = trainer.train()

    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

    torch.save(best_model_wts, f"model_distillation_pth/best_model_weights_{student_name}_{distiller_name}.pth")

if __name__ == "__main__":
    args = parse_args()
    main(selected_student=0, selected_teacher=0, selected_distiller=args.selected_distiller, batch_size=64, temp=1)