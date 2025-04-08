import os
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from utils.transforms import get_transform

def get_loaders(data_path, batch_size=64, val_split=0.2, colour=True):
    transform = get_transform(colour=colour)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes
