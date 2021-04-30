import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import os
import pickle

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

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

def get_dataloader(root_dir, is_train, batch_size, workers, resolution=32, classes=1000):
    normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078],
                                     std=[0.2146, 0.2104, 0.2138])
    transformations = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ] if is_train else [
        transforms.ToTensor(),
        normalize,
    ]
    trans = transforms.Compose(transformations)
    dataset = SmallImagenet(root=root_dir, size=resolution, train=is_train, transform=trans,
                            classes=range(classes))
    shuffle = True if is_train else False
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True)
    return loader


class SmallImagenet(VisionDataset):
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target