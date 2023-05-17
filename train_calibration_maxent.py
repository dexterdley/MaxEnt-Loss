#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from utils import load_pretrained, save_checkpoint, evaluate, convert_to_onehot, smooth_one_hot

from Net.resnet import ResNet18, ResNet50
from Net.densenet import DenseNet121
from Net.resnet_tinyimagenet import ResNet18_TinyimgNet, ResNet50_TinyimgNet

import argparse,random
from torch import linalg
import os

from losses import *
import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
from synthesize_images import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='Synthetic noisy label ratio')
    parser.add_argument('--arch', type=str, default="res50", help='Model backbone architecture')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--seed', type=int, default=1, help='Random Seed')

    parser.add_argument('--alpha', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal factor')
    parser.add_argument('--constraints', type=int, default=0, help='Max Ent mode constraints 1-Mu 2-Variance 3-Poly')

    return parser.parse_args()

class ModelFeature(nn.Module):
    def __init__(self, dataset='CIFAR', num_classes=10, arch ='res18'):
        super(ModelFeature, self).__init__()

        if arch == 'densenet121':
            self.model = DenseNet121(num_classes=num_classes)

        if arch == 'res18' and dataset == 'tinyimagenet':
            self.model = ResNet18_TinyimgNet(num_classes=num_classes)

        elif arch == 'res18' and dataset != 'tinyimagenet':
            self.model = ResNet18(num_classes=num_classes)

        if arch == 'res50' and dataset == 'tinyimagenet':
            print("OK")
            self.model = ResNet50_TinyimgNet(num_classes=num_classes)

        elif arch == 'res50' and dataset != 'tinyimagenet':
            self.model = ResNet50(num_classes=num_classes)

    def forward(self, x):

        x = self.model(x)

        return x

def add_noise(array, noise_ratio, num_classes):
    corrupted_len = int(noise_ratio * len(array) )
    corrupted_labels = random.choices( range(num_classes), k=corrupted_len)
    idx = random.choices( range(len(array) ), k=corrupted_len)
    array[idx] = torch.tensor(corrupted_labels).to(device)

    return array

def run_training():

    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    expt_name = args.expt_name + '_' + str(args.seed)
    dataset = expt_name.split('_')[0]

    train_dataset, val_dataset, test_dataset, num_classes = prepare_dataset(dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    
    split = torch.zeros(num_classes)
    print('Counting prior')
    for _, y in train_loader:    
        onehot = convert_to_onehot(y , num_classes)
        split += onehot.sum(0)
    ratio = split/split.sum()
    print(ratio)
    
    #pdb.set_trace()
    #Loss and optimiser
    model = ModelFeature(dataset=dataset, num_classes=num_classes, arch=args.arch) 
    params = model.parameters()
    model.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=5e-4)                     
    else:
        raise ValueError("Optimizer not supported.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train the network
    print('Running:', expt_name, 'Batch size:', batch_size)

    best_val_accuracy = 0

    if args.constraints == 0:
        criterion = FocalLoss(gamma=0) #Regular CE loss

    elif args.constraints == 1:
        criterion = FocalLoss(gamma=args.gamma)

    elif args.constraints == 2:
        criterion = InvFocalLoss(gamma=args.gamma)

    elif args.constraints == 3:
        criterion = AUAvULoss(beta=3.0)

    elif args.constraints == 4:
        criterion = SoftAUAvULoss(num_classes=num_classes)

    elif args.constraints == 5:
        criterion = PolyLoss()
    
    elif args.constraints == 6:
        criterion = MaxEntLoss(ratio=ratio, constraints=args.constraints, num_classes=num_classes, gamma=args.gamma)

    elif args.constraints == 7:
        criterion = MaxEntLoss(ratio=ratio, constraints=args.constraints, num_classes=num_classes, gamma=args.gamma)

    elif args.constraints == 8:
        criterion = MaxEntLoss(ratio=ratio, constraints=args.constraints, num_classes=num_classes, gamma=args.gamma)

    for epoch in range(num_epochs):

        losses = []
        CE_losses = []

        feature_norm_list = []

        num_correct = 0
        num_samples = 0
        model.train()
        
        for batch_idx, (data, GT_classes) in tqdm(enumerate(train_loader)):
            
            # Get data to cuda if possible
            data = data.to(device) #images
            GT_classes = GT_classes.to(device)

            if args.noise_ratio > 0:
                GT_classes = add_noise(GT_classes, args.noise_ratio, num_classes)

            targets_var = convert_to_onehot(GT_classes, num_classes).to(device) #convert classes to onehot representation
            
            if args.alpha > 0: # Apply label smoothing
                targets_var = smooth_one_hot(targets_var, num_classes, args.alpha)

            # forward
            logits = model(data)
            class_pred = F.softmax(logits, dim=1)
        
            ''' Computes the classification loss (Cross entropy loss) '''
            loss, CE_loss = criterion(class_pred, targets_var)
            CE_losses.append(CE_loss.item())

            feature_norm = linalg.norm(logits, 2).cpu().detach().numpy()
            optimizer.zero_grad() # backward

            if args.constraints == 3 and epoch<20:
                CE_loss.backward()
                optimizer.step()

            else:
                loss.backward() # gradient descent or adam step
                optimizer.step()

            losses.append(loss.item())
            feature_norm_list.append(feature_norm)

            num_correct += torch.sum(torch.eq(class_pred.argmax(1), GT_classes)).item()
            num_samples += class_pred.size(0)

        #pdb.set_trace()
        scheduler.step()
        print('Epoch {}, Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))

        train_accuracy = num_correct/num_samples
        f_norm = sum(feature_norm_list)/len(feature_norm_list)

        train_stats = ",\t".join([
                f'Epoch: {epoch} TRAIN Total loss: {sum(losses)/len(losses):.3f}',
                f'Acc: {train_accuracy:.3f}',
                f"CE loss: {sum(CE_losses)/len(CE_losses):.3f}",
                ])
        print(train_stats)
        
        #Before temperature scaling
        before_ce_val_loss, scores, val_accuracy, val_F1, ECE, MCE, NLL, brier, adaece, cece, KS_error = evaluate(val_loader, model, num_classes, T=1)
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
                f"KS_error: {KS_error:.3f}",
                ])
        print(val_stats)

        if not os.path.exists('./logs/'  + expt_name.split("_")[0]):
            os.makedirs('./logs/' + expt_name.split("_")[0])

        with open('./logs/' + expt_name.split("_")[0] + '/' + expt_name + '_scores.txt', "a") as f:
            print(train_accuracy, val_accuracy, val_F1, ECE, MCE, NLL, brier, f_norm, adaece, cece, KS_error,  file=f)
            
        if val_accuracy > best_val_accuracy and epoch != 0:
                best_val_accuracy = val_accuracy
                checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
                save_checkpoint(checkpoint, 'best', expt_name)

    #Evaluate the best performing model
    best_model_path = './models/' + expt_name + '/' + 'best_' + expt_name + '.pth.tar'
    best_model = load_pretrained(model, best_model_path)

    _, test_scores, test_accuracy, test_F1, test_ECE, test_MCE, test_NLL, test_brier, test_adaece, test_cece, KS_error = evaluate(test_loader, model, num_classes, T=1)
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
                f"KS_error: {KS_error:.3f}",
                ])

    with open(expt_name.split("_")[0] + '_test_scores.txt', "a") as f:
        print(test_stats, file=f)
    print(test_stats)

if __name__ == "__main__":                    
    run_training()