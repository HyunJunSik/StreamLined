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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logging(teacher_name, student_name, distiller_name, temp):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.shutdown()  
    logging.getLogger().handlers.clear() 
    log_filename = f"./SDD_train_log/experiment_cifar-100_{now}_{teacher_name}_{student_name}_{distiller_name}.log"
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
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return optimizer
    
    def train(self):
        epoch_length = 240
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
            
            if test_acc1 > best_acc:
                best_acc = test_acc1
                best_model_wts = copy.deepcopy(self.distiller.module.student.state_dict())
            self.distiller.module.student.load_state_dict(best_model_wts)
        return best_model_wts
              
if __package__ is None:
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Distiller import distiller
    from Distiller.SDD import SDD_DKD

    from SDD_Models import vgg, wrn, shufflenet_v1, mobilenetv2, shufflenet_v2, resnet
    from train import load_dataset_multiapply, load_dataset
    
def main(selected_student, selected_teacher, selected_distiller, batch_size=64, temp=4): 
    temperature = temp
    train_loader, val_loader =  load_dataset(bz=batch_size)
    model_student, student_name = [
        # resnet.resnet8(num_classes=100),
        resnet.resnet32(num_classes=100),
        resnet.resnet8x4(num_classes=100),
        resnet.resnet20(num_classes=100),
        shufflenet_v1.ShuffleV1(num_classes=100),
        shufflenet_v2.ShuffleV2(num_classes=100),
        vgg.vgg8(num_classes=100),
        wrn.wrn_40_1(num_classes=100),
        wrn.wrn_16_2(num_classes=100),
        mobilenetv2.mobile_half(num_classes=100),
        ][selected_student]
    

    model_teacher, teacher_name = [
        # resnet.resnet32(num_classes=100), 
        resnet.resnet32x4(num_classes=100), 
        resnet.resnet56(num_classes=100),
        resnet.resnet110(num_classes=100),
        vgg.vgg13(num_classes=100),
        wrn.wrn_40_2(num_classes=100),
        ][selected_teacher]
    
    model_teacher.load_state_dict(load_teacher_param(teacher_name))
        
    distiller = [
        SDD_DKD(model_student, model_teacher),
        ][selected_distiller]
    distiller_name = [
        "SDD",
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

    torch.save(best_model_wts, f"SDD_pth/best_model_weights_{student_name}_{distiller_name}.pth")
    


def load_teacher_param(model_name):
    model_state_dict_path = f"SDD_model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path)
    return model_state_dict

if __name__ == "__main__":
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    print(f"device : {device}")
