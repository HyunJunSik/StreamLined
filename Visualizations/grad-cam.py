import os
import sys
from os import path
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib.pyplot as plt
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug


if __package__ is None:
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from imagenet_models import imagenet_resnet, mobilenetv2
else:
    from ..imagenet_models import imagenet_resnet, mobilenetv2

def load_param(model_name):
    model_state_dict_path = f"./../utils/imagenet_model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path, map_location=torch.device('cpu'))
    return model_state_dict

def load_sample(image_name):
    image_name = image_name + ".jpg"
    image = Image.open(image_name).convert("RGB")
    image = image.resize((224, 224))
    
    # 저장
    name = image_name + "_resized.jpg"
    image.save(name)
    # 임시 블록
    image_array = np.array(image).astype(np.float32) / 255.0 # [0, 1]정규화
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image, image_array

def grad_cam(model, model_name, tensor, label):
    # 마지막 conv layer: ResNet은 layer4가 가장 마지막 feature map임
    if "resnet" in model_name:
        target_layers = [model.layer4[-1]]
    else: # mobilenet
        target_layers = [model.model[12]]
    
    # Grad-CAM++ 객체 생성
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    
    return grayscale_cam

def main(model, model_name, distiller_name):
    img_dict = {"ruddy" : 17, "Sorrel" : 339, "Cocker" : 160}
    visualization_list = []
    
    for k, v in img_dict.items():
        img_tensor, img_arr = load_sample("./" + k)
        # Grad-CAM++ 실행
        CAM = grad_cam(model, model_name, img_tensor.to(device), v)
        visualization = show_cam_on_image(img_arr, CAM, use_rgb=True)    
        visualization_list.append(visualization)
      
    combined_visualization = np.concatenate(visualization_list, axis=0)  
    plt.imsave(model_name + "_" + distiller_name + ".jpg", combined_visualization)
        
        

    

if __name__=="__main__":
    path = os.getcwd()
    parent_path = os.path.dirname(path)
    parser = argparse.ArgumentParser(description="grad-cam++ model and distiller selection")
    parser.add_argument("--model", type=str, required=True, choices=["res18", "res34", "mb"])
    parser.add_argument("--distiller", type=str, choices=["MLKD", "RKD", "ReviewKD", "SKD"], default="none")
    args = parser.parse_args()
    
    distiller = None if args.distiller == "none" else args.distiller
    model, model_name = imagenet_resnet.resnet18(pretrained=True) if args.model == "res18" else mobilenetv2.mobilenetv2()

    if distiller == None:
        if args.model == "mb":
            PATH = os.path.join(parent_path, "utils/imagenet_model_pth/best_model_weights_mobilenetv2.pth")
        model_state_dict = torch.load(PATH, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
    else:
        if args.distiller == "MLKD":
            PATH = os.path.join(parent_path, "utils/model_distillation_pth/best_model_weights_mobilenetv2_MLKD_align_3_5.pth")
        elif args.distiller == "RKD":
            PATH = os.path.join(parent_path, "utils/model_distillation_pth/[RKD] best_model_weights_mobilenetv1_RKD.pth")
        elif args.distiller == "ReviewKD":
            PATH = os.path.join(parent_path, "utils/model_distillation_pth/[ReviewKD] best_model_weights_mobilenetv1_ReviewKD.pth")
        elif args.distiller == "SKD":
            PATH = os.path.join(parent_path, "utils/model_distillation_pth/[SKD] best_model_weights_mobilenetv1_SKD.pth")
        model_state_dict = torch.load(PATH, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
    model.to(device)
    main(model, model_name, args.distiller)
    
