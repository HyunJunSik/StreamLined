# Streamlined Knowledge Distillation

This repository is the official implementation of [Streamlined Knowledge Distillation]

![propose](https://github.com/user-attachments/assets/2d9501d4-81a2-44bc-b0bd-ee74b1e9b255)

We refactored our code based on [MDistiller](https://github.com/megvii-research/mdistiller).
While the internal structure (e.g., the mdistiller folder) may differ from the original repository, the functionality remains consistent.

## Directory Setup
Before training, please create the following directories under the CIFAR-100/Utils folder
```
CIFAR-100/Utils/
├── model_train_log
├── model_distillation_pth
├── distiller_train_log
├── SDD_model_pth
├── SDD_train_log
```
```
ImageNet/Utils/
├── model_train_log
├── model_distillation_pth
├── distiller_train_log
├── SDD_model_pth
├── SDD_train_log
```
These folders are required to stor training logs and model checkpoints.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

Trained models are saved under the following directories:
```
- CIFAR-100/Utils/model_distillation_pth
- CIFAR-100/Utils/SDD_model_pth
```
Please configure paths accordingly before running training.
For CIFAR-100:
```train
python CIFAR-100/Utils/distillation_train.py --selected_student 3 --selected_teacher 1 --selected_distiller 2
```
For ImageNet:
python ImageNet/Utils/distillation_train.py --selected_distiller 0

## Test

Test files exist in each dataset's "Utils" folder
you can evaluate trained model by test.py

## Results

Our model achieves the following performance on :

### [Image Classification on CIFAR-100 under homogeneous architectures]

| Teacher    | ResNet56 | ResNet110 | ResNet32x4 | WRN-40-2     | WRN-40-2     | VGG13 |
|------------|----------|-----------|------------|--------------|--------------|--------|
|            | 72.40    | 74.31     | 79.34      | 75.59        | 75.59        | 74.64  |
| Student    | ResNet20 | ResNet32 | ResNet8x4   | WRN-16-2     | WRN-40-1     | VGG13  |
|            | 65.79    | 67.92     | 72.48      | 71.12        | 69.54        | 68.19  |
| **Feature-based KD** |          |           |            |              |              |        |
| FitNet     | 69.64    | 72.35     | 75.26      | 74.46        | 73.28        | 70.09  |
| RKD        | 70.86    | 73.81     | 74.93      | 74.73        | 74.30        | 71.61  |
| CRD        | 67.28    | 71.97     | 74.16      | 74.22        | 72.44        | 72.74  |
| OFD        | 68.77    | 71.40     | 73.88      | 72.31        | 73.26        | 74.21  |
| ReviewKD   | 69.54    | 71.12     | 75.71      | 73.92        | 74.50        | 72.29  |
| **Logit-based KD**   |          |           |            |              |              |        |
| KD         | 71.74    | 74.01     | 74.75      | 76.04        | 74.52        | 74.08  |
| CLKD       | 65.74    | 69.81     | 70.19      | 72.89        | 71.89        | 72.46  |
| DKD        | 71.17    | 74.12     | 76.51      | 76.41        | 75.33        | 74.41  |
| MLKD       | 72.21    | 74.24     | 75.59      | **76.83**        | 74.78        | 74.25  |
| SDD        | 69.42    | 72.78     | 74.40      | 75.01        | 72.53        | 72.29  |
| **Ours**   | **72.50**| **74.84** | **78.33**  | 76.60    | **76.04**    | **75.75** |


## Acknowledgement

We would like to thank the contributors of MDistiller for their excellent work, which served as the foundation for our implementation.
