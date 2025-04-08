import os
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from utils.dataset import get_char_dataset, get_img_dataset

def get_char_loaders(data_path, batch_size=64, val_split=0.2, colour=True):
    dataset = get_char_dataset(data_path, colour=True)

    if val_split == 0:
        train_set = dataset
        val_set = None
    else:
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    return train_loader, val_loader, dataset.classes


def get_img_loaders(data_path, batch_size=64, val_split=0.2, colour=True):
    dataset = get_img_dataset(data_path, colour=True)

    if val_split == 0:
        train_set = dataset
        val_set = None
    else:
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    return train_loader, val_loader, len(dataset.vocab)