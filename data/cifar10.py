from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=test_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
