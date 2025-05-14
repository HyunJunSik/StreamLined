# Streamlined Knowledge Distillation

This repository is the official implementation of [Streamlined Knowledge Distillation]

![propose](https://github.com/user-attachments/assets/f9fae55f-be82-4a20-a61e-5cdd3a2238ce)

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


## Evaluation

To evaluate my model on CIFAR-100, run:

```eval
python test.py
```

## Results

Our model achieves the following performance on :

### [Image Classification on CIFAR-100 under homogeneous architectures]

| Teacher    | ResNet56 | ResNet110 | ResNet32x4 | WRN-40-2     | WRN-40-2     | VGG13 |
|------------|----------|-----------|------------|--------------|--------------|--------|
|            | 71.22    | 73.06     | 77.34      | 75.24        | 75.24        | 72.00  |
| Student    | ResNet20 | ResNet32 | ResNet8x4   | WRN-16-2     | WRN-40-1     | VGG13  |
|            | 65.79    | 67.92     | 72.48      | 71.12        | 69.54        | 68.19  |
| **Feature-based KD** |          |           |            |              |              |        |
| FitNet     | 68.51    | 71.27     | 73.79      | 72.51        | 71.85        | 71.75  |
| RKD        | 70.86    | 73.81     | 74.93      | 74.73        | 74.30        | 72.05  |
| CRD        | 68.68    | 72.02     | 73.66      | 74.22        | 72.44        | 72.74  |
| OFD        | 67.15    | 70.28     | 74.01      | 71.95        | 72.21        | 73.75  |
| ReviewKD   | 69.14    | 71.58     | 76.14      | 75.13        | 73.82        | 73.26  |
| **Logit-based KD**   |          |           |            |              |              |        |
| KD         | 69.34    | 73.27     | 75.45      | 75.25        | 75.12        | 71.35  |
| CLKD       | 65.74    | 69.81     | 70.19      | 72.89        | 71.89        | 72.46  |
| DKD        | 69.73    | 72.83     | 76.65      | 75.28        | 74.40        | 71.81  |
| MLKD       | 71.05    | 73.75     | 76.75      | 75.88        | 75.45        | 72.42  |
| SDD        | 68.22    | 70.97     | 74.83      | 73.33        | 72.94        | 73.09  |
| **Ours**   | **71.34**| **74.26** | **77.21**  | **76.42**    | **75.56**    | **74.10** |


## Acknowledgement

We would like to thank the contributors of MDistiller for their excellent work, which served as the foundation for our implementation.
