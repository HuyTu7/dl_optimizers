import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch


def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )

    return train_dataset


def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize_transform()
    ])

    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )

    return val_dataset


def data_loader(data_dir, batch_size=256, workers=2, pin_memory=True):
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size / 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader