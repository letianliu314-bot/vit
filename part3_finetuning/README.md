# Part 3: Downstream Fine-tuning

## Overview

This part reproduces the fine-tuning experiments from the ViT paper, evaluating transfer learning performance of **ViT-B/32** pretrained on **ImageNet-21k** across four downstream datasets.

## Key Findings

Our implementation achieves **better accuracy** with **substantially fewer training steps** compared to the original paper:

| Dataset | Original Acc (%) | Our Acc (%) | Original Steps | Our Steps |
|---------|------------------|-------------|----------------|-----------|
| CIFAR-10 | 98.79 | **98.90** | 10,000 | **800** |
| CIFAR-100 | 91.97 | **92.72** | 10,000 | **800** |
| Oxford Flowers-102 | 99.11 | **99.51** | 500 | **100** |
| Oxford-IIIT-Pets | 93.02 | **94.11** | 500 | **100** |
| **Average** | 95.72 | **96.31** | 5,250 | **450** |

- **Accuracy Gain**: +0.59% average improvement
- **Efficiency**: ~12× faster convergence (5,250 → 450 steps)

## Why Our Results Are Better

The performance gap stems from pretrained weights. The original paper (2020) used basic regularization (weight decay, dropout, label smoothing). We use **`timm`'s pretrained weights with AugReg**—state-of-the-art augmentation and regularization techniques—yielding better features that converge faster.

## Implementation Details

### Model Architecture
- **Base Model**: ViT-B/32 (patch size 32×32, 12 layers, 768 hidden dim, 12 heads)
- **Pretrained**: ImageNet-21k (via `timm.create_model('vit_base_patch32_224', pretrained=True)`)
- **Resolution**: 384×384 (upscaled from 224×224 during fine-tuning)

### Training Hyperparameters

| Parameter | CIFAR-10/100 | Flowers/Pets |
|-----------|--------------|--------------|
| Total Steps | 10,000 | 500 |
| Batch Size | 512 | 512 |
| Learning Rate | 0.01 | 0.01 |
| Optimizer | SGD (momentum=0.9, weight_decay=0.0) |
| Scheduler | Cosine Annealing |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | BF16 (GradScaler) |

> **Note on Training Steps**: The step counts (10,000 for CIFAR, 500 for Flowers/Pets) are set to **match the original paper** for fair comparison. However, due to `timm`'s superior AugReg pretrained weights, our model achieves **better accuracy in far fewer steps** (e.g., ~800 steps for CIFAR, ~100 steps for Flowers/Pets). You can reduce `TOTAL_STEPS` accordingly to save training time while maintaining or even improving performance.

### Critical Implementation Notes

1. **Head Initialization**: The new classification head must be initialized to **zero** (as specified in the paper):
   ```python
   nn.init.zeros_(model.head.weight)
   nn.init.zeros_(model.head.bias)
   ```

2. **Data Preprocessing**: Google ViT official normalization to [-1, 1]:
   ```python
   vit_mean, vit_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
   ```

3. **Transform Pipeline**:
   - Resize to `int(384 / 0.875)` ≈ 438 (using BICUBIC interpolation)
   - RandomCrop(384) for training / CenterCrop(384) for evaluation
   - RandomHorizontalFlip

4. **Step-based Training**: Unlike epoch-based training, ViT fine-tuning uses a fixed number of optimization steps with an infinite data iterator:
   ```python
   def get_infinite_batches(dataloader):
       while True:
           for batch in dataloader:
               yield batch
   ```

## Notebooks

| Notebook | Dataset | Classes | Steps |
|----------|---------|---------|-------|
| `vit-a100-vitb32-cifar10.ipynb` | CIFAR-10 | 10 | 10,000 |
| `vit-a100-vitb32-cifar100.ipynb` | CIFAR-100 | 100 | 10,000 |
| `vit-a100-vitb32-flowers102.ipynb` | Oxford Flowers-102 | 102 | 500 |
| `vit-a100-vitb32-pets.ipynb` | Oxford-IIIT-Pets | 37 | 500 |

## Hardware Requirements

- **GPU**: A100 40GB (or equivalent with ~24GB+ VRAM)
- **Memory**: Batch size 512 with BF16 mixed precision
- **Storage**: ~2GB for datasets

## Dependencies

```bash
pip install timm torch torchvision scipy
```

## Usage

Each notebook is self-contained. Simply run the cells in order:

1. Install dependencies
2. Set HF endpoint for faster model downloads (China region)
3. Execute training script

Example for CIFAR-10:
```python
# Configuration
DATASET_NAME = "CIFAR10"
NUM_CLASSES = 10
TOTAL_STEPS = 10000  # Paper-aligned; can reduce to ~800 for similar/better results
BASE_LR = 0.01
RESOLUTION = 384

# Run training
main()
```

## References

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., *et al.* **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**. ICLR 2021.
2. Steiner, A., Kolesnikov, A., Zhai, X., *et al.* **How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers**. TMLR 2022.
