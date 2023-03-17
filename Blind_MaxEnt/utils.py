#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:28:29 2020

@author: Anonymous cat
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
import pdb
import torch.nn as nn

from ECE import *
import torch
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

criterion = F.binary_cross_entropy_with_logits

def mse(A, B, class_weights):
    
    if class_weights is None:
        
        y = torch.square(A - B)
        
    else:
        
        y = torch.square(A - B) * class_weights
    
    out = torch.mean(y)
    return out

def mae(pred, y, class_weights):
    
    B, k = pred.shape #batch by number of classes
    
    zeros = torch.zeros(pred.shape).to(device)
    
    if class_weights is None:
        
        out = torch.max(zeros, (pred - y))
        
    else:
        
        out = torch.max(zeros, (pred - y)) * class_weights.reshape(B, 1)
    
    out = torch.sum(out, dim=1) 
    
    #pdb.set_trace()
    return torch.mean(out)

def negative_log_likelihood(pred, y, class_weights):

    ce_loss = y * torch.log(pred.clamp(min=1e-6)) 
    
    if class_weights is None:
        
        ce_loss = -torch.sum(ce_loss, dim=1)
        
    else:
        
        ce_loss = -torch.sum(ce_loss, dim=1)  * class_weights #changed 
        
    #pdb.set_trace()
    return torch.mean(ce_loss)


def get_top_classes(vec, rank, top_k):
    new_vec = np.zeros(rank.shape)

    for i in range(top_k):
    
        top_idx = np.where(i == rank)
        new_vec[top_idx] = vec[top_idx]
        
    #pdb.set_trace()    
    return new_vec

def convert_to_onehot(array, num_classes=8):
    
    B = len(array) #batch size
    
    out = np.zeros((B, num_classes))
    out[range(B), array.to(cpu)] = 1
    
    return torch.FloatTensor(out)

def evaluate(loader, model, num_classes, T):
    """
    Parameters
    ----------
    loader : Dataloader
        Validation dataset
    model : TYPE
        DESCRIPTION.
    
    Using only 3-channel RGB model
    
    Returns
    -------
    RMSE and CCC values for both Arousal and Valence predictions,
    Classification report
    """
    
    model.eval()
    cpu = torch.device('cpu')
    
    with torch.no_grad():
        
        ce_losses = []

        pred_probs = []
        GT_all = []
        
        num_correct_classes = 0
        num_samples_classes = 0

        labels_list = []
        predictions_list = []
        confidence_vals_list = []

        NLL_error_list = []
        brier_error_list = []

        for count, (x, labels) in enumerate(loader):
            
            x = x.to(device = device)
            labels = labels.to(device = device)

            #forward
            logits = model(x)
            class_pred = F.softmax(logits/T, dim=1)
            classes = torch.argmax(class_pred, dim=1)

            pred_probs.append(class_pred)
            GT_all.append(labels)
            
            onehot = convert_to_onehot(labels, num_classes=num_classes).to(device) #convert classes to onehot representation
            y_idx = np.where(onehot.cpu() ==1) 
            
            CE_loss = negative_log_likelihood(class_pred, onehot, None)
            ce_losses.append(CE_loss.item())

            compare_array = torch.eq(classes, labels )

            wrong_idx = torch.where(compare_array == 0) # index of misclassified samples
            wrong_probs = class_pred[wrong_idx] #predicted probabilites of misclassified samples
            
            NLL_error = -torch.sum(onehot[wrong_idx] * torch.log(wrong_probs.clamp(min=1e-6)), dim=1).mean() # measure the NLL of misclassified samples
            brier_error = torch.square(onehot[wrong_idx] - wrong_probs).mean() # measure the brier loss of misclassified samples
            
            NLL_error_list.append(NLL_error.item() )
            brier_error_list.append(brier_error.item() )

            num_correct_classes += torch.sum(torch.eq(classes, labels )).item()
            num_samples_classes += len(labels)

            confidence_vals, predictions = torch.max(class_pred, dim=1)
            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().detach().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().detach().numpy().tolist())

        pred_probs = torch.cat(pred_probs)
        pred_all = torch.argmax(pred_probs, 1)
        GT_all = torch.cat(GT_all)
        
        scores = metrics.classification_report(GT_all.to(cpu), pred_all.to(cpu), digits = 4)
        F1 = metrics.f1_score(GT_all.to(cpu), pred_all.to(cpu), average='macro')
        accuracy = num_correct_classes/num_samples_classes

        avg_ce_loss = sum(ce_losses)/len(ce_losses)

        #compute ECE here
        ECE = expected_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)
        MCE = maximum_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)
        NLL = sum(NLL_error_list)/len(NLL_error_list)
        brier = sum(brier_error_list)/len(brier_error_list)
        adaece = adaptive_expected_calibration_error(confidence_vals_list, predictions_list, labels_list ).item()
        cece = classwise_calibration_error(pred_probs.cpu(), labels_list, num_classes ).item()

        #pdb.set_trace()
        return avg_ce_loss, scores, accuracy, F1, ECE, MCE, NLL, brier, adaece, cece

def save_checkpoint(state, epoch, expt_name):
    print("==> Checkpoint saved")
    
    if not os.path.exists('./models/' + expt_name):
        os.makedirs('./models/' + expt_name)
        
    outfile = './models/' + expt_name + '/' + str(epoch) + '_' + expt_name + '.pth.tar'
    torch.save(state, outfile)
    
def load_checkpoint(model, optimizer, weight_file):
    print("==> Loading Checkpoint: " + weight_file)
    
    #weight_file = r'checkpoint.pth.tar'
    if torch.cuda.is_available() == False:
        checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(weight_file)
        
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])