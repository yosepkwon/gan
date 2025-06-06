# dataloader.py
import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomCelebA(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.image_dir = os.path.join(root, 'celeba', 'img_align_celeba')
        self.attr_path = os.path.join(root, 'celeba', 'Anno', 'list_attr_celeba.txt')
        self.transform = transform
        self.selected_attrs = ['Blond_Hair', 'Male', 'Young']

        lines = open(self.attr_path, 'r').readlines()
        self.filenames = []
        self.labels = []
        self.attr2idx = {attr_name: idx for idx, attr_name in enumerate(lines[1].split())}

        for line in lines[2:]:
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = [(int(values[self.attr2idx[attr]]) + 1) // 2 for attr in self.selected_attrs]
            self.filenames.append(filename)
            self.labels.append(label)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = torch.FloatTensor(self.labels[index])

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.filenames)



class RafdDataset(Dataset):
    def __init__(self, root, transform=None):
        self.label_path = os.path.join(root, 'train_labels.csv')
        self.root = os.path.join(root, 'DATASET', 'train')  # train/감정번호/파일.jpg
        self.transform = transform

        df = pd.read_csv(self.label_path)
        self.images = df['image'].tolist()
        self.labels = df['label'].tolist()
        self.num_classes = 8  # RaFD 감정 개수

    def _one_hot(self, label):
        vec = [0] * self.num_classes
        vec[label] = 1
        return vec

    def __getitem__(self, index):
        img_name = self.images[index]
        label = self._one_hot(self.labels[index])
        image_path = os.path.join(self.root, str(self.labels[index]), img_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)

