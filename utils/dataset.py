from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import torch
import cv2
from PIL import Image
from utils.preprocessing import crop_image, seq_to_chars

VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"

def get_char_dataset(data_path, colour, resize_to):
    return datasets.ImageFolder(root=data_path, transform=get_transform(colour, resize_to))


def get_img_dataset(data_path, colour):
    return CaptchaDataset(data_path, colour)

def get_test_dataset(data_path, colour, resize_to, train_path):
    return TestDataset(data_path, colour, resize_to, train_path)

def get_transform(colour, resize_to):
    transform_list = []

    if resize_to:
        transform_list.append(transforms.Resize(resize_to)) 

    if colour:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else:
        transform_list.extend([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    return transforms.Compose(transform_list)


class TestDataset(Dataset):
    def __init__(self, folder_path, colour, resize_to, train_path):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = get_transform(colour)
        self.seq_to_chars = seq_to_chars
        self.colour = colour
        self.resize_to = resize_to
        self.vocab = VOCAB
        self.idx_to_char = datasets.ImageFolder(root=train_path).classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label = filename.split("-")[0]
        
        image_path = os.path.join(self.folder_path, filename)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        char_images_np = self.seq_to_chars(image, self.colour, resize_to=self.resize_to)

        char_images = []
        for c in char_images_np:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            char_images.append(self.transform(Image.fromarray(c)))

        return char_images, label


class CaptchaDataset(Dataset):
    def __init__(self, folder_path, colour):
        self.folder_path = folder_path
        self.colour = colour
        self.image_files = os.listdir(folder_path)
        self.vocab = VOCAB
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.vocab)}  # 0 is CTC blank
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label_str = filename.split("-")[0]
        label = torch.tensor([self.char_to_idx[c] for c in label_str], dtype=torch.long)        

        image_path = os.path.join(self.folder_path, filename)
        image = cv2.imread(image_path)
        image = crop_image(image)
        
        if not self.colour:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 50))
            image = image.astype("float32") / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (200, 50))
            image = image.astype("float32") / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, label, len(label)

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, label_lengths

