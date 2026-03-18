import argparse
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import BATCH_SIZE, DEPTH, EMBED_DIM, HEADS, NUM_CLASSES, PATCH_SIZE
from data.cifar10 import get_cifar10_dataloaders
from models.vit import VisionTransformer


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "multistep":
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1,
        )

    def lr_lambda(epoch: int) -> float:
        if args.epochs <= 1:
            return 1.0

        min_factor = args.min_lr / args.lr
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))

        cosine_total = max(1, args.epochs - args.warmup_epochs)
        cosine_step = min(epoch - args.warmup_epochs, cosine_total)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_total))
        return min_factor + (1.0 - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "multistep"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = VisionTransformer(
        img_size=32,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=HEADS,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    run_name = f"vit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_acc = (outputs.argmax(dim=1) == labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += batch_acc
            global_step += 1

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/acc_step", batch_acc, global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}", lr=f"{current_lr:.2e}")

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
        writer.add_scalar("train/acc_epoch", avg_acc, epoch + 1)
        writer.add_scalar("train/lr_epoch", current_lr, epoch + 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f}, acc={avg_acc:.4f}, lr={current_lr:.2e}"
        )

        if scheduler is not None:
            scheduler.step()

    ckpt_path = os.path.join(args.save_dir, "vit_last.pth")
    torch.save(model.state_dict(), ckpt_path)
    writer.close()
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
