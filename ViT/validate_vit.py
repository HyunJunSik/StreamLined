import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import register_model
from timm.layers import trunc_normal_
from os import path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from imagenet_train import get_imagenet_dataloaders_strong

def test(model, criterion, test_loader):
    
    model.eval()
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    dataset_len = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

def load_sample(image_name):
    '''
    shark, cock, frog
    '''
    image_name = image_name + ".JPEG"
    image = Image.open(image_name).convert("RGB")
    image = image.resize((224, 224))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image

def main(model):
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    img = load_sample("./../visualization/frog").to(device)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    
    output = model(img)
    print(f"label : {output.argmax().item()}")
    
    # _, test_loader, num_data = get_imagenet_dataloaders_strong(512, 128, 32)
    # test_loss, test_acc1, test_acc5 = test(model, criterion, test_loader)
    
    # print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")

if __name__ == "__main__":
    model = timm.create_model('regnety_160.tv2_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    main(model)
