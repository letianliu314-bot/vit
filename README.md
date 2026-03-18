# An Image is Worth 16×16 Words: ViT Reproduction Study

This project reproduces **Vision Transformer (ViT)** as described in *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (ICLR 2021), validating the paper's core claims about inductive bias, internal mechanisms, and transfer learning.

## Overview

Vision Transformer (ViT) treats images as patch sequences, lacking CNN's inductive biases (locality, translation equivariance). The original paper showed ViT underperforms on small datasets but excels at scale (JFT-300M, ImageNet-21k).

**This study validates:**
1. The small-data gap on CIFAR-10 (ViT vs ResNet18)
2. Internal mechanisms via attention visualization
3. Transfer learning effectiveness via fine-tuning

## Project Structure

```
vit/
|
|-- part1_inductive_bias/     # ViT vs ResNet18 on CIFAR-10 from scratch
|-- part2_visualization/       # Attention rollout, position embeddings, etc.
|-- part3_finetuning/          # Downstream task fine-tuning (ViT-B/32)
|
`-- README.md
```

---

## Part 1: Inductive Bias Validation

**Location:** `part1_inductive_bias/`

Validates the small-data regime: ViT splits images into 16×16 patches, embeds each as a token, and processes the sequence through Transformer encoder blocks. Compared to ResNet-18, ViT lacks spatial inductive biases.

**Results (CIFAR-10 from scratch, 300 epochs):**

| Model | Validation Accuracy |
|-------|---------------------|
| ViT (ours) | 78.84% |
| ResNet-18 | **86.15%** |

**Conclusion:** On limited data, CNNs outperform plain ViT due to stronger inductive biases.

**Quick Start:**
```bash
cd part1_inductive_bias
pip install torch torchvision tqdm tensorboard
python train_vit.py      # Train ViT
python train_resnet.py   # Train ResNet-18 baseline
```

---

## Part 2: Model Internal Mechanism Visualization

**Location:** `part2_visualization/`

Visualizes how pretrained ViT-L/16 processes images through:

1. **Position Embedding Analysis** – Cosine similarity reveals 2D topology encoding
2. **Patch Embedding Filters** – PCA shows Gabor-like bases (similar to early CNNs)
3. **Attention Rollout** – Aggregates layer-wise attention to highlight salient regions
4. **Attention Distance & Entropy** – Deeper layers show larger receptive fields

**Key Insight:** ViT learns spatial structure internally despite no explicit inductive bias.

---

## Part 3: Downstream Fine-tuning

**Location:** `part3_finetuning/`

Reproduces fine-tuning of **ViT-B/32** (pretrained on ImageNet-21k) on four datasets:

| Dataset | Original Acc (%) | Our Acc (%) | Original Steps | Our Steps |
|---------|------------------|-------------|----------------|-----------|
| CIFAR-10 | 98.79 | **98.90** | 10,000 | **800** |
| CIFAR-100 | 91.97 | **92.72** | 10,000 | **800** |
| Oxford Flowers-102 | 99.11 | **99.51** | 500 | **100** |
| Oxford-IIIT-Pets | 93.02 | **94.11** | 500 | **100** |
| **Average** | 95.72 | **96.31** | 5,250 | **450** |

**Key Findings:**
- **+0.59%** average accuracy gain
- **~12× faster** convergence (5,250 → 450 steps)
- Better results due to `timm`'s AugReg pretrained weights

**Quick Start:**
```bash
cd part3_finetuning
pip install timm torch torchvision scipy

# Run individual notebooks
jupyter notebook vit-a100-vitb32-cifar10.ipynb
```

---

## References

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., *et al.* **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**. ICLR 2021.

2. Abnar, S., & Zuidema, W. **Quantifying Attention Flow in Transformers**. ACL 2020.

3. Chefer, H., Gur, S., & Wolf, L. **Transformer Interpretability Beyond Attention Visualization**. CVPR 2021.

4. Steiner, A., Kolesnikov, A., Zhai, X., *et al.* **How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers**. TMLR 2022.

---

## Code Repository

**GitHub:** https://github.com/letianliu314-bot/vit

**Group:** COMP7404 Group 1
