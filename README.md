# Streamlined Knowledge Distillation

This repository is the official implementation of [Streamlined Knowledge Distillation]

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet]
|------------|----------|-----------|------------|--------------|--------------|--------|
| Teacher    | ResNet56 | ResNet110 | ResNet32x4 | WRN-40-2     | WRN-40-2     | VGG13 |
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
|------------|----------|-----------|------------|--------------|--------------|--------|


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
