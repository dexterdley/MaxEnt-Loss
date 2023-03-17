import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image

sz = 32
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#T0 Most basic image transformation
T0 = [
    transforms.Resize(size=(sz,sz),interpolation=2),
    transforms.ToTensor(), #3*H*W, [0, 1]
    normalize] # normalize with mean/std

T1 = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=30,translate=(.1, .1),scale=(1.0, 1.25),resample=Image.BILINEAR),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.Resize(size=(sz,sz),interpolation=2),
    transforms.ToTensor(), #3*H*W, [0, 1]
    normalize]

T2 = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=30,translate=(.1, .1),scale=(1.0, 1.25),resample=Image.BILINEAR),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.Resize(size=(sz,sz),interpolation=2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(90),
    transforms.ToTensor(), #3*H*W, [0, 1]
    normalize]

T3 = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=30,translate=(.1, .1),scale=(1.0, 1.25),resample=Image.BILINEAR),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.Resize(size=(sz,sz),interpolation=2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(90),
    transforms.RandomAdjustSharpness(0),
    transforms.RandomPerspective(),
    transforms.ToTensor(), #3*H*W, [0, 1]
    normalize]