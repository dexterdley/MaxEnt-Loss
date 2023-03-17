#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:30:17 2020

@author: Anonymous Cat
"""
# imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import copy

import pandas as pd
import torch
import torchvision

import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils import load_checkpoint, save_checkpoint, evaluate, convert_to_onehot
import pdb
import argparse,random
from torch import linalg
import os
import sklearn
from sklearn.model_selection import train_test_split

from maxent_newton_solver import *

#torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
from synthesize_images import T0, T1, T2, T3


def smooth_one_hot(onehot, num_classes: int, epsilon=0.01):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    
    out = (1 - epsilon) * onehot + epsilon/num_classes
    
    return out

class MaxEntLoss(nn.Module):
    def __init__(self, ratio, constraints, gamma=2, num_classes=10, eps=1e-7):
        super(MaxEntLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps

        self.constraints = constraints
        x = torch.tensor(range(num_classes), dtype=float)
        
        target_mu = torch.sum(ratio * x)
        target_var = torch.sum(ratio * x.pow(2)) - target_mu.pow(2)

        self.lam_1 = solve_mean_lagrange(x, target_mu)
        self.lam_2 = solve_var_lagrange(x, target_mu, target_var)
        
        self.lam_1a, self.lam_2a = solve_multiple_lagrange(x, target_mu, target_var)
        self.lam_1a, self.lam_2a = self.lam_1a.to(device), self.lam_2a.to(device)

        self.x = torch.tensor(range(num_classes), dtype=float).to(device)
        self.target_mu = target_mu.to(device)
        self.target_var = target_var.to(device)

        print("Expected mean and Lam1:", self.target_mu.item(), self.lam_1.item())
        print("Expected variance and Lam2:", self.target_var.item(), self.lam_2.item())
        print("Combined Lambda:", self.lam_1a.item(), self.lam_2a.item())
       
    def forward(self, p, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """

        batch_sz = y.shape[0]
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))

        mu = torch.sum(p * self.x, dim=1)
        var = torch.sum(p * self.x.pow(2), dim=1) - mu.pow(2)

        # Focal loss
        if self.gamma > 0:
            focal = (1 - p).pow(self.gamma)
            loss *= focal

        if self.constraints == 1: #Exponential Distribution (Mean constraint)
        
            mu_loss = torch.abs(mu - self.target_mu)
            loss = -torch.sum(loss, dim=1) + self.lam_1 * mu_loss

        elif self.constraints == 2: #Gaussian/Normal Distribution (Variance constraint)

            var_loss = torch.abs(var - self.target_var)
            loss = -torch.sum(loss, dim=1) + self.lam_2 * var_loss

        elif self.constraints == 3: #Poly loss
        
            epsilon = -1
            poly = loss + epsilon * (1 - y*p)
            loss = -torch.sum(poly, dim=1)

        elif self.constraints == 4: #Combine multiple constraints
        
            mu_loss = torch.abs(mu - self.target_mu)
            var_loss = torch.abs(var - self.target_var)

            loss = -torch.sum(loss, dim=1) + self.lam_1a * mu_loss + self.lam_2a * var_loss

        elif self.constraints == 5: #Inv focal
            
            inv_focal = -(1 + p).pow(self.gamma)
            loss *= inv_focal

        else:
            loss = -torch.sum(loss, dim=1)

        #pdb.set_trace()
        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for sgd.')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='Synthetic noisy label ratio')
    parser.add_argument('--arch', type=str, default=0, help='Model backbone architecture')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--shift_ood', type=str, default='T1', help='Type of distribution shift')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')

    parser.add_argument('--alpha', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal factor')
    parser.add_argument('--constraints', type=int, default=0, help='Max Ent mode constraints 1-Mu 2-Variance 3-Poly')

    return parser.parse_args()

''' 
End of helper functions
'''
class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes=10, arch ='res18'):
        super(Res18Feature, self).__init__()
        if arch == 'res18':
            resnet  = torchvision.models.resnet18(pretrained)
        elif arch == 'res50':
            resnet  = torchvision.models.resnet50(pretrained)
        elif arch == 'densenet121':
            resnet = torchvision.models.densenet121(pretrained)

        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1
        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x8

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
     
        x = self.fc(x)
        return x #classification output and pre-softmax features

def load_pretrained(res18, model_path):
    pretrained = torch.load(model_path)
    pretrained_state_dict = pretrained['state_dict']
    model_state_dict = res18.state_dict()
    loaded_keys = 0
    total_keys = 0
    for key in pretrained_state_dict:
        if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
            pass
        else:    
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys+=1
            if key in model_state_dict:
                loaded_keys+=1
    print("Loaded params num:", loaded_keys)
    print("Total params num:", total_keys)
    res18.load_state_dict(model_state_dict, strict = False)

    return res18

def run_training():

    num_classes = 10
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    torch.manual_seed(args.seed)

    expt_name = args.expt_name + '_' + str(args.seed)
    dataset = expt_name.split('_')[0]

    easy_transformations = T0

    if args.shift_ood == 'T0':
        hard_transformations = T0

    elif args.shift_ood == 'T1':
        hard_transformations = T1

    elif args.shift_ood == 'T2':
        hard_transformations = T2

    elif args.shift_ood == 'T3':
        hard_transformations = T3

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose(easy_transformations) )
        val_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose(hard_transformations) )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_indices, val_indices = sklearn.model_selection.train_test_split(indices, test_size=0.2)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = datasets.CIFAR10('../data', train=False, download=True,  transform=transforms.Compose(hard_transformations) )

    if dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transforms.Compose(easy_transformations) )
        val_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transforms.Compose(hard_transformations) )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_indices, val_indices = sklearn.model_selection.train_test_split(indices, test_size=0.2)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = datasets.CIFAR100('../data', train=False, download=True,  transform=transforms.Compose(hard_transformations) )
        num_classes = 100

    elif dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + easy_transformations))
        val_dataset = datasets.FashionMNIST('../data', train=True, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + hard_transformations))

        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_indices, val_indices = sklearn.model_selection.train_test_split(indices, test_size=0.2)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = datasets.FashionMNIST('../data', train=False, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + hard_transformations))


    elif dataset == 'mnist':
        train_dataset = datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + easy_transformations))
        val_dataset = datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + hard_transformations))

        num_train = len(train_dataset)
        indices = list(range(num_train))
        train_indices, val_indices = sklearn.model_selection.train_test_split(indices, test_size=0.2)

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        test_dataset = datasets.MNIST('../data', train=False, download=True, 
                       transform=transforms.Compose( [transforms.Grayscale(num_output_channels=3)] + hard_transformations))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    

    split = torch.zeros(num_classes)

    print('Counting prior')
    for batch_idx, (x, y) in tqdm(enumerate(train_loader)):    
        onehot = convert_to_onehot(y , num_classes)
        split += onehot.sum(0)

    ratio = split/split.sum()
    print(ratio)
    criterion = MaxEntLoss(ratio=ratio, constraints=args.constraints, num_classes=num_classes, gamma=args.gamma)

    #pdb.set_trace()
    #Loss and optimiser
    res18 = Res18Feature(pretrained=True, num_classes=num_classes, arch=args.arch) 
    params = res18.parameters()
    res18.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-5)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-5)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-5)                           
    else:
        raise ValueError("Optimizer not supported.")

    # Train the network
    print('Running:', expt_name, 'Batch size:', batch_size)

    best_F1 = 0
    best_val_accuracy = 0
    best_val_nll = 10 ** 7
    temperature_list = [1.25, 1.5, 1.75, 2.0]
    best_T = 1

    for epoch in range(num_epochs):

        losses = []
        R_losses = []
        CE_losses = []

        feature_norm_list = []
        weight_norm_list = []

        num_correct = 0
        num_samples = 0
    
        for batch_idx, (data, GT_classes) in tqdm(enumerate(train_loader)):
            
            # Get data to cuda if possible
            data = data.to(device) #images
            GT_classes = GT_classes.to(device)

            targets_var = convert_to_onehot(GT_classes, num_classes).to(device) #convert classes to onehot representation
            
            if args.alpha > 0: #label smoothing
                targets_var = smooth_one_hot(targets_var, num_classes, args.alpha)

            # forward
            features = res18(data)
            class_pred = F.softmax(features, dim=1)
        
            ''' Computes the classification loss (Cross entropy loss) '''            
            loss, CE_loss = criterion(class_pred, targets_var)
            CE_losses.append(CE_loss.item())

            feature_norm = linalg.norm(features, 2).cpu().detach().numpy()
            weight_norm = linalg.norm(res18.fc.weight, 2).cpu().detach().numpy()

            optimizer.zero_grad() # backward
            loss.backward() # gradient descent or adam step
            optimizer.step()

            losses.append(loss.item())
            feature_norm_list.append(feature_norm)
            weight_norm_list.append(weight_norm)

            num_correct += torch.sum(torch.eq(class_pred.argmax(1), GT_classes)).item()
            num_samples += class_pred.size(0)

        #break
        #pdb.set_trace()
        res18.train()
        print('Epoch {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))

        train_accuracy = num_correct/num_samples
        f_norm = sum(feature_norm_list)/len(feature_norm_list)
        w_norm = sum(weight_norm_list)/len(weight_norm_list)

        train_stats = ",\t".join([
                f'Epoch: {epoch} TRAIN Total loss: {sum(losses)/len(losses):.3f}',
                f'Acc: {train_accuracy:.3f}',
                f"CE loss: {sum(CE_losses)/len(CE_losses):.3f}",
                ])
        print(train_stats)
        
        #Before temperature scaling
        before_ce_val_loss, scores, val_accuracy, val_F1, ECE, MCE, NLL, brier, adaece, cece = evaluate(val_loader, res18, num_classes, T=1)
        val_stats = ",\t".join([
                f'Before T CE loss: {before_ce_val_loss:.3f}',
                f"Val Acc: {val_accuracy:.3f}",
                f"Val F1: {val_F1:.3f}",
                f"ECE: {ECE:.3f}",
                f"MCE: {MCE:.3f}",
                f"NLL: {NLL:.3f}",
                f"Brier: {brier:.3f}",
                f"AdaECE: {adaece:.3f}",
                f"CCE: {cece:.3f}",
                ])
        print(val_stats)

        if not os.path.exists('./logs/' + args.shift_ood + '_' + expt_name.split("_")[0]):
            os.makedirs('./logs/' + args.shift_ood + '_' + expt_name.split("_")[0])

        with open('./logs/' + args.shift_ood + '_' + expt_name.split("_")[0] + '/' + expt_name + '_scores.txt', "a") as f:
            print(train_accuracy, val_accuracy, val_F1, ECE, MCE, NLL, brier, f_norm, adaece, cece,  file=f)
            
        #After temperature scaling
        for T in tqdm(temperature_list):
            
            after_ce_val_loss, scores, val_accuracy, val_F1, ECE, MCE, NLL, brier, adaece, cece = evaluate(val_loader, res18, num_classes, T)
            val_stats = ",\t".join([
                    f'After T CE loss: {after_ce_val_loss:.3f}',
                    f"Val Acc: {val_accuracy:.3f}",
                    f"Val F1: {val_F1:.3f}",
                    f"ECE: {ECE:.3f}",
                    f"MCE: {MCE:.3f}",
                    f"NLL: {NLL:.3f}",
                    f"Brier: {brier:.3f}",
                    f"AdaECE: {adaece:.3f}",
                    f"CCE: {cece:.3f}",
                    ])
            print(val_stats)

            if after_ce_val_loss < best_val_nll:
                best_T = T
                best_val_nll = after_ce_val_loss

        if val_accuracy > best_val_accuracy and epoch != 0:
                best_val_accuracy = val_accuracy
                checkpoint = {'state_dict': res18.state_dict(),'optimizer': optimizer.state_dict()}
                save_checkpoint(checkpoint, 'best', expt_name)

    #Evaluate the best performing model
    best_model_path = './models/' + expt_name + '/' + 'best_' + expt_name + '.pth.tar'
    best_res18 = load_pretrained(res18, best_model_path)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    _, test_scores, test_accuracy, test_F1, test_ECE, test_MCE, test_NLL, test_brier, test_adaece, test_cece = evaluate(test_loader, res18, num_classes, T=1)
    # test on test set
    test_stats = ",\t".join([
                f'Test Acc: {test_accuracy:.3f}',
                f'Test F1: {test_F1:.3f}',
                f"ECE: {test_ECE:.3f}",
                f"MCE: {test_MCE:.3f}",
                f"NLL: {test_NLL:.3f}",
                f"Brier: {test_brier:.3f}",
                f"AdaECE: {test_adaece:.3f}",
                f"CCE: {test_cece:.3f}",
                ])

    with open(expt_name.split("_")[0] + '_test_scores.txt', "a") as f:
        print(test_stats, file=f)
    print(test_stats)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    _, test_scores, test_accuracy, test_F1, test_ECE, test_MCE, test_NLL, test_brier, test_adaece, test_cece = evaluate(test_loader, res18, num_classes, best_T)
    # test on test set with temperature scaling
    test_stats = ",\t".join([
                f'Test Acc: {test_accuracy:.3f}',
                f'Test F1: {test_F1:.3f}',
                f"ECE: {test_ECE:.3f}",
                f"MCE: {test_MCE:.3f}",
                f"NLL: {test_NLL:.3f}",
                f"Brier: {test_brier:.3f}",
                f"AdaECE: {test_adaece:.3f}",
                f"CCE: {test_cece:.3f}",
                f"Temperature: {best_T:.3f}",
                ])

    with open(expt_name.split("_")[0] + '_test_scores.txt', "a") as f:
        print(test_stats, file=f)



if __name__ == "__main__":                    
    run_training()