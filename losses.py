import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn.metrics import auc
import torch.nn.functional as F
import pdb

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

from maxent_newton_solver import *

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps

    def forward(self, p, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))

        if self.gamma > 0:
            # Focal loss
            focal = (1 - p).pow(self.gamma)
            loss *= focal

        loss = -torch.sum(loss, dim=1)
        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class InvFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(InvFocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        
    def forward(self, p, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """
        
        # Basic CE computation
        loss = y * torch.log(p.clamp(min=self.eps))
            
        inv_focal = -(1 + p).pow(self.gamma)
        loss *= inv_focal

        loss = torch.sum(loss, dim=1)
        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class PolyLoss(nn.Module):
    def __init__(self, epsilon=-1, eps=1e-8):
        super(PolyLoss, self).__init__()

        self.eps = eps
        self.epsilon = epsilon
        
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

        poly = loss + self.epsilon * (1 - y*p)
        loss = -torch.sum(poly, dim=1)

        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

class MaxEntLoss(nn.Module):
    def __init__(self, ratio, constraints, gamma=2, num_classes=10, eps=1e-8):
        super(MaxEntLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes
        self.ratio = ratio.to(device)

        self.constraints = constraints
        x = torch.tensor(range(num_classes), dtype=float)
        
        global_target_mu = torch.sum(ratio * x)
        global_target_var = torch.sum(ratio * x.pow(2))

        local_target_mu = torch.sum(torch.eye(num_classes) * x, dim=1)
        local_target_var = torch.sum(torch.eye(num_classes) * x.pow(2), dim=1)

        mix_mu = (global_target_mu + local_target_mu)/2 #combine
        mix_var = (global_target_var + local_target_var)/2

        mult_global_target_var = torch.sum(ratio * (x - mix_mu).pow(2))
        mult_local_target_var = torch.sum(torch.eye(num_classes) * (x- mix_mu).pow(2), dim=1)

        mix_mult_var = (mult_global_target_var + mult_local_target_var)/2

        self.local_lam_1 = {}
        for i in range(num_classes):
            self.local_lam_1[i] = solve_mean_lagrange(x, mix_mu[i])

        self.local_lam_2 = {}
        for i in range(num_classes):
            self.local_lam_2[i] = solve_var_lagrange(x, mix_var[i])

        self.local_lam_mult = {}
        for i in range(num_classes):
            self.local_lam_mult[i] = solve_multiple_lagrange(x, mix_mu[i], mix_mult_var[i])

        print(self.local_lam_mult)

        self.x = torch.tensor(range(num_classes), dtype=float).to(device)
        self.target_mu = global_target_mu.to(device)
        self.target_var = global_target_var.to(device)
        self.target_mult_var = self.target_var - self.target_mu.pow(2)

        #pdb.set_trace()
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
        var = torch.sum(p * self.x.pow(2), dim=1)
        mult_var = torch.sum(p * (self.x.repeat(batch_sz,1) - mu.unsqueeze(-1)).pow(2), dim=1)

        local_mu = torch.sum(y*self.x, dim=1)
        local_var = torch.sum(y*self.x.pow(2), dim=1)
        local_mult_var = local_var - local_mu.pow(2)

        # Focal loss
        if self.gamma > 0:
            focal = (1 - p).pow(self.gamma)
            loss *= focal

        if self.constraints == 6: #Exponential Distribution (Mean constraint)
            mu_loss = torch.abs(mu - self.target_mu)
            local_mu_loss = torch.abs(mu - local_mu)

            lam = torch.tensor([self.local_lam_1[j.cpu().item()] for j in y.argmax(1) ]).to(device)
            loss = -torch.sum(loss, dim=1) + lam * (mu_loss + local_mu_loss)

        elif self.constraints == 7: #Gaussian/Normal Distribution (Variance constraint)
            var_loss = torch.abs(var - self.target_var)
            local_var_loss = torch.abs(var - local_var)

            lam = torch.tensor([self.local_lam_2[j.cpu().item()] for j in y.argmax(1)]).to(device)
            loss = -torch.sum(loss, dim=1) + lam * (var_loss + local_var_loss)

        elif self.constraints == 8: #Combine multiple constraints
            mu_loss = torch.abs(mu - self.target_mu)
            local_mu_loss = torch.abs(mu - local_mu)
                        
            var_loss = torch.abs(mult_var - self.target_mult_var)
            local_var_loss = torch.abs(mult_var - local_mult_var)

            lam = torch.tensor([self.local_lam_mult[j.cpu().item()] for j in y.argmax(1)]).to(device)
            loss = -torch.sum(loss, dim=1) + lam[:,0] * (mu_loss + local_mu_loss) + lam[:,1] * (var_loss + local_var_loss)

        return loss.mean(), -torch.sum(y * torch.log(p.clamp(min=self.eps)), dim=1).mean()

def my_auc(x, y):
    direction =1 
    dx = torch.diff(x)

    if torch.any(dx < 0):
        if torch.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing : {}.".format(x))

    return direction * torch.trapz(y, x)

class AUAvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is probabilities from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]

    The reference code is from: https://github.com/IntelLabs/AVUC/blob/main/src/avuc_loss.py
    """
    def __init__(self, beta=1):
        super(AUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -torch.sum(prob * torch.log(prob.clamp(self.eps)), dim=1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def forward(self, probs, y, type=0):

        confidences, predictions = torch.max(probs, 1)
        labels = torch.argmax(y)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        #th_list = np.linspace(0, 1, 21)
        th_list = torch.linspace(0, 1, 21, requires_grad=True).to(device)
        umin, umax = torch.min(unc), torch.max(unc)

        avu_list = []
        unc_list = []

        for t in th_list:
            unc_th = umin + (torch.tensor(t, device=labels.device) * (umax - umin))
            
            n_ac = torch.zeros(1, device=labels.device)
            n_ic = torch.zeros(1, device=labels.device)
            n_au = torch.zeros(1, device=labels.device)
            n_iu = torch.zeros(1, device=labels.device)
            
            #Use masks and logic operators to compute the 4 differentiable proxies
            n_ac_mask = torch.logical_and(labels == predictions, unc <= unc_th)
            n_ac = torch.sum( confidences[n_ac_mask] * (1 - torch.tanh(unc[n_ac_mask]) ) )

            n_au_mask = torch.logical_and(labels == predictions, unc > unc_th)
            n_au = torch.sum( confidences[n_au_mask] * torch.tanh(unc[n_au_mask]) )

            n_ic_mask = torch.logical_and(labels != predictions, unc <= unc_th)
            n_ic = torch.sum( (1 - confidences[n_ic_mask] ) * (1 - torch.tanh(unc[n_ic_mask]) ) )

            n_iu_mask = torch.logical_and(labels != predictions, unc > unc_th)
            n_iu = torch.sum( (1 - confidences[n_iu_mask] ) *  torch.tanh(unc[n_iu_mask]) )

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        auc_avu = my_auc(th_list, torch.stack(avu_list))
        CE_loss = -torch.sum(y * torch.log(probs.clamp(min=self.eps)), dim=1).mean()
        avu_loss = -self.beta * torch.log(auc_avu + self.eps) + CE_loss
        return avu_loss, CE_loss

class SoftAUAvULoss(nn.Module):
    """
    Calculates Soft Accuracy vs Uncertainty Loss of a model.
    The input to this loss is probabilites from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]

    The reference codes are from: 
    1.) https://github.com/IntelLabs/AVUC/blob/main/src/avuc_loss.py
    2.) https://github.com/google/uncertainty-baselines/blob/main/experimental/caltrain/secondary_losses.py
    """
    def __init__(self, beta=1, num_classes=10):
        super(SoftAUAvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10
        self.entmax = torch.log(torch.tensor(num_classes) ).to(device)
        self.focal = FocalLoss(gamma=1)

    def entropy(self, prob):
        return -torch.sum(prob * torch.log(prob.clamp(self.eps)), dim=1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def soft_T(self, e, temp=0.01, theta=0.1):
        numerator = e * (1 - theta)
        denominator = (1 - e) * theta
        frac = numerator/denominator.clamp(self.eps)
        v = 1/temp * torch.log(frac.clamp(self.eps))
        #print(v.mean())
        return torch.sigmoid(v)

    def forward(self, probs, y, type=0):

        confidences, predictions = torch.max(probs, 1)
        labels = torch.argmax(y)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        th_list = torch.linspace(0, 1, 21, requires_grad=True).to(device)
        umin, umax = torch.min(unc), torch.max(unc)

        avu_list = []
        unc_list = []

        #auc_avu = torch.ones(1, device=labels.device, requires_grad=True)

        for t in th_list:
            unc_th = umin + (torch.tensor(t) * (umax - umin))
            
            n_ac = torch.zeros(1, device=labels.device)
            n_ic = torch.zeros(1, device=labels.device)
            n_au = torch.zeros(1, device=labels.device)
            n_iu = torch.zeros(1, device=labels.device)
            
            #Use masks and logic operators to compute the 4 differentiable proxies
            n_ac_mask = torch.logical_and(labels == predictions, unc <= unc_th)
            n_ac = torch.sum( (1 - self.soft_T(unc[n_ac_mask]/self.entmax)) * (1 - torch.tanh(unc[n_ac_mask]) ) )

            n_au_mask = torch.logical_and(labels == predictions, unc > unc_th)
            n_au = torch.sum( self.soft_T(unc[n_au_mask]/self.entmax) * torch.tanh(unc[n_au_mask]) )

            n_ic_mask = torch.logical_and(labels != predictions, unc <= unc_th)
            n_ic = torch.sum( (1 - self.soft_T(unc[n_ic_mask]/self.entmax) ) * (1 - torch.tanh(unc[n_ic_mask]) ) )

            n_iu_mask = torch.logical_and(labels != predictions, unc > unc_th)
            n_iu = torch.sum( self.soft_T(unc[n_iu_mask]/self.entmax) *  torch.tanh(unc[n_iu_mask]) )

            AvU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
            avu_list.append(AvU)
            unc_list.append(unc_th)

        auc_avu = my_auc(th_list, torch.stack(avu_list))
        focal_loss, CE_loss = self.focal(probs, y)

        Savu_loss = -self.beta * torch.log(auc_avu.clamp(min=self.eps)) + focal_loss        
        return Savu_loss, CE_loss