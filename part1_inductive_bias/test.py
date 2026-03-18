import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnet18

from data.cifar10 import get_cifar10_dataloaders
from models.vit import vit_tiny_patch4_32
from utils import AverageMeter, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits, targets), bs)

    return loss_meter.avg, acc_meter.avg


def build_model(model_name: str, num_classes: int):
    if model_name == "vit":
        return vit_tiny_patch4_32(num_classes=num_classes)
    if model_name == "resnet18":
        return resnet18(num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["vit", "resnet18"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=10).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
