import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
from tqdm import tqdm
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import numpy as np
import time
import logging
import copy
import sys
from os import path
from datetime import datetime


data_folder = os.path.join(path.dirname(path.abspath(__file__)), '../imagenet')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Invert(img, _):
    return ImageOps.invert(img)

def Identity(img, v):
    return img

def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)

def Rotate(img, v):
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    # v =-v
    return img.rotate(v)

def Sharpness(img, v):
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v):
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateXabs(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateYabs(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def Solarize(img, v):
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)

def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    
    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Cutout(img, v):
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img
    
    v = v * img.size[0]
    return CutoutAbs(img, v)


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

class MultipleApply:
    '''
    Apply a list of transformations to an image and get multiple transformed images
    
    Args : transforms (list or tuple) : list of transformations
    '''
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        return [t(image) for t in self.transforms]

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m # [0, 30] in fixmatch, deprecated
        self.augment_list = augment_list()
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img

def get_imagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform_weak = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform_strong = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    # train_transform = MultipleApply([train_transform_weak, train_transform_strong])
    return train_transform_weak

def get_imagenet_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform

def get_imagenet_val_loader(val_batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    test_transform = get_imagenet_test_transform(mean, std)
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=32, pin_memory=True)
    return test_loader

def get_imagenet_dataloaders_strong(batch_size, val_batch_size, num_workers, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set =  ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data

def train(model, criterion, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    with tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False) as pbar:
        for idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            optimizer.zero_grad()

            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            # loss.item()은 loss값을 스칼라로 반환
            # _, predicted = outputs.max(1) outputs.max(1)은 각 입력 샘플에 대해 가장 큰 값과 해당 인덱스 반환
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            total_loss += loss.item() * batch_size
            total_acc1 += acc1.item() * batch_size
            total_acc5 += acc5.item() * batch_size
            
    dataset_len = len(train_loader.dataset) 
    train_loss = total_loss / dataset_len
    train_acc1 = total_acc1 / dataset_len
    train_acc5 = total_acc5 / dataset_len
        
    return train_loss, train_acc1, train_acc5

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

def main(model, model_name):
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"device : {device}")
    
    train_loader, test_loader, num_data = get_imagenet_dataloaders_strong(512, 128, 32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0001)
    model.to(device)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    logging.basicConfig(filename=f"./imagenet_train_log/imagenet_{now}_{model_name}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    
    best_acc = 0.0
    epoch_length = 100
    
    for epoch in range(epoch_length):
        if epoch in [30, 60, 90]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        logging.info(f"Epoch: {epoch + 1}/{epoch_length}")
        print(f"epochs : {epoch + 1} / {epoch_length}")
        train_loss, train_acc1, train_acc5 = train(model, criterion, train_loader, optimizer)

        test_loss, test_acc1, test_acc5 = test(model, criterion, test_loader)

        if test_acc1 > best_acc:
            best_acc = test_acc1
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        
        print(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        logging.info(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        
        print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
        logging.info(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"./checkpoint/{model_name}_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch' : epoch + 1,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'best_acc' : best_acc,
            }, checkpoint_path)
            print(f"Checkpoint saved")
        
        
    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    
    torch.save(best_model_wts, f"./imagenet_model_pth/best_model_weights_{model_name}.pth")
    print(f"Learning Time : {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from models import resnet, mobilenetv2
    else:
        from ..models import resnet, mobilenetv2
    
    # resnet-32, resnet-18, resnet-50, mobilenet-v2
    model, model_name = mobilenetv2.mobilenetv2(num_classes=1000)
    main(model, model_name)

# checkpoint_epoch = 50  # 예: 50 epoch에서 재개
# checkpoint_path = f"imagenet_model_pth/checkpoint_epoch_{checkpoint_epoch}.pth"

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']

# print(f"Resuming training from epoch {start_epoch}")
