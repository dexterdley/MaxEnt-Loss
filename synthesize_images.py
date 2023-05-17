import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import random
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
import os
from tiny_img import download_tinyImg200
import pandas as pd

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

class TinyImageNet_dataset(Dataset): 
    
    def __init__(self, root_path, transform=None):

        self.root_path = root_path
        self.transform = transform

        self.dataframe = pd.read_csv("tiny_imagenet_train.csv")
        self.data = self.dataframe["0"]
        self.labels = self.dataframe["labels"]

    def __getitem__(self, index):

        img_path = self.data[index]
        image = Image.open(self.root_path + str(img_path)).convert('RGB')
        target = self.labels[index]

        if self.transform != None:
            image = self.transform(image)

        return [image, target]

    def __len__(self):
        return len(self.data)

class TinyImageNet_C_dataset(Dataset): 
    
    def __init__(self, root_path, severity=1, transform=None):

        self.root_path = root_path
        self.transform = transform
        self.severity = severity

        if self.severity == 0:
            self.dataframe = pd.read_csv("tiny_imagenet_0.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]
            self.root_path = 'tiny-imagenet-200/val/images/'

        elif self.severity == 1:
            self.dataframe = pd.read_csv("tiny_imagenet_1.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]

        elif self.severity == 2:
            self.dataframe = pd.read_csv("tiny_imagenet_2.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]

        elif self.severity == 3:
            self.dataframe = pd.read_csv("tiny_imagenet_3.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]

        elif self.severity == 4:
            self.dataframe = pd.read_csv("tiny_imagenet_4.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]

        elif self.severity == 5:
            self.dataframe = pd.read_csv("tiny_imagenet_5.csv")
            self.data = self.dataframe["0"]
            self.labels = self.dataframe["labels"]

    def __getitem__(self, index):

        img_path = self.data[index]
        image = Image.open(self.root_path + str(img_path)).convert('RGB')
        target = self.labels[index]

        if self.transform != None:
            image = self.transform(image)

        return [image, target]

    def __len__(self):
        return len(self.data)

train_transformations = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(), #3*H*W, [0, 1]
    normalize]

val_transformations = [transforms.ToTensor(), #3*H*W, [0, 1]
    normalize]

def prepare_dataset(dataset):

    if dataset == 'cifar10':
        
        dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose(train_transformations) )
        test_dataset = datasets.CIFAR10('../data', train=False, download=True,  transform=transforms.Compose(val_transformations) )
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
        num_classes = 10

    elif dataset == 'cifar100':

        dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transforms.Compose(train_transformations) )
        test_dataset = datasets.CIFAR100('../data', train=False, download=True,  transform=transforms.Compose(val_transformations) )
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
        num_classes = 100

    elif dataset == 'tinyimagenet':
        
        if not os.path.exists('./tiny-imagenet-200/'):
            download_tinyImg200('.')

        dataset = TinyImageNet_dataset('tiny-imagenet-200/train/', transform=transforms.Compose(train_transformations))
        #test_dataset = datasets.ImageFolder('tiny-imagenet-200/val', transform=transforms.Compose(val_transformations))
        test_dataset = TinyImageNet_C_dataset(root_path='tiny-imagenet-200/val', severity=0, transform=transforms.Compose(val_transformations) )

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [90000, 10000])
        num_classes = 200

    return train_dataset, val_dataset, test_dataset, num_classes