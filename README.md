# ViT Reproduction (CIFAR-10)

This project is a simple reproduction of Vision Transformer (ViT) and a ResNet-18 baseline on CIFAR-10.

## Project Structure

```
vit-reproduction/
|
|-- models/
|   |-- vit.py
|   |-- resnet.py
|
|-- data/
|   |-- cifar10.py
|
|-- train_vit.py
|-- train_resnet.py
|-- test.py
|
|-- utils.py
|-- config.py
|
`-- README.md
```

## Setup

Install dependencies:

```bash
pip install torch torchvision tqdm tensorboard
```

## Train ViT

```bash
python train_vit.py
```

## Train ResNet-18 Baseline

```bash
python train_resnet.py
```

## Test

Test ViT checkpoint:

```bash
python test.py --model vit --ckpt checkpoints/vit_last.pth
```

Test ResNet-18 checkpoint:

```bash
python test.py --model resnet18 --ckpt checkpoints/resnet_last.pth
```

## Analysis

Our result reproduces the small-data regime behavior described in *An Image is Worth 16x16 Words*. This result is consistent with the findings in the original paper: when training from scratch on a limited dataset, CNN-based models are typically stronger than plain ViT baselines.

In this project, both models are trained and evaluated on CIFAR-10, with standard train/test protocol (50k training images and 10k test images). Training uses the same data source and evaluation is reported on the held-out test split. Under this setup, the final performance is approximately **ResNet-18: 86%** and **ViT: 78%**, which verifies the core trend from the paper in the small-data regime: **ResNet > ViT**.

Why this happens is also clear from the behavior we observe. ResNet has a built-in locality inductive bias from convolutions, so it can learn useful edge/texture patterns efficiently even with limited supervision. ViT has weaker spatial priors and must learn these relationships from data, which is harder when the dataset is small and no large-scale pretraining (for example, JFT-300M-level pretraining) is available. As a result, ViT in this reproduction is more optimization-sensitive and reaches a lower final accuracy under the same data budget.
