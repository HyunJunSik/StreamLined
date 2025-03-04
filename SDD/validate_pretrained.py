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
import argparse
from os import path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from models import resnet, mobilenetv2
    from Distiller import distiller
    from Distiller.KD import KD
    from Distiller.DKD import DKD
    from Distiller.CLKD import CLKD
    from Distiller.Proposed import GCKD_normalized_masking
    from Distiller.MLKD import MLKD_align_5, MLKD_align_4, MLKD_align_3
    # from Distiller.FitNet import FitNet
    # from Distiller.RKD import RKD
    # from Distiller.ReviewKD import ReviewKD
    from imagenet_train import get_imagenet_dataloaders_strong
else:
    from ..models import resnet, mobilenetv2

def test(model, criterion, test_loader):
    
    model.eval()
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    dataset_len = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            # _, predicted = outputs.max(1)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc1 += acc1.item() * batch_size
            total_acc5 += acc5.item() * batch_size
    val_loss = total_loss / dataset_len
    val_acc1 = total_acc1 / dataset_len
    val_acc5 = total_acc5 / dataset_len
    return val_loss, val_acc1, val_acc5

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


def main(model, model_name):
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    _, test_loader, num_data = get_imagenet_dataloaders_strong(512, 128, 32)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    logging.basicConfig(filename=f"./model_performance/imagenet_{now}_{model_name}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    
    test_loss, test_acc1, test_acc5 = test(model, criterion, test_loader)
    
    print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
    logging.info(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")

    
if __name__ == "__main__":
    chosen_model = [
        resnet.resnet18(pretrained=True, num_classes=1000),
        resnet.resnet34(pretrained=True, num_classes=1000),
        # imagenet_resnet.resnet50(pretrained=True, num_classes=1000),
        mobilenetv2.mobilenetv2(num_classes=1000),
    ]
    model_idx = 0
    model, model_name = chosen_model[model_idx]
    if model_idx == 2: # MobileNetV2
        model_state_dict_path = f"./imagenet_model_pth/best_model_weights_mobilenetv2.pth"
        model_state_dict = torch.load(model_state_dict_path)
        model.load_state_dict(model_state_dict)
    main(model, model_name)
        