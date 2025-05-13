import torch
import torch.nn as nn
import sys
from os import path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, model_name, criterion, test_loader):
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = test_loss / total
        epoch_acc = correct / total * 100
        with open("./model_performance/model_performance.txt", "a") as f:
            f.write(f"{model_name}-Test | Loss:%.4f Acc: %.2f%% (%s/%s)\n"
            %(epoch_loss, epoch_acc, correct, total))
    return epoch_loss, epoch_acc

def main(model, model_name, data_loader):
    model.to(device)
    
    model_state_dict_path = f"model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path)
    
    model.load_state_dict(model_state_dict)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(model, model_name, criterion, data_loader)

if __name__=="__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from Models import wrn, shufflenet_v1, resnet, vgg
        from train import load_dataset
    else:
        from ..Models import wrn, shufflenet_v1, resnet, vgg
        from .train import load_dataset
    
    model, model_name = resnet.resnet32x4(num_classes=100)
    train_loader, test_loader = load_dataset()
    
    main(model, model_name, test_loader)

    