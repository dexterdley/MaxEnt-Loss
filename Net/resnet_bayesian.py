'''
Bayesian ResNet for CIFAR10.
ResNet architecture ref:
https://arxiv.org/abs/1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Module, Parameter
import math


__all__ = [
    'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'
]

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0

class LinearVariational(Module):
    def __init__(self,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 in_features,
                 out_features,
                 bias=True):

        super(LinearVariational, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        self.posterior_rho_init = posterior_rho_init,  # variance of weight --> sigma = log (1 + exp(rho))
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu', torch.Tensor(out_features))
        else:
            self.register_buffer('prior_bias_mu', None)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)

        self.mu_weight.data.normal_(std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.mu_bias.data.normal_(std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        sigma_p = torch.tensor(sigma_p)
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.sum()

    def forward(self, input):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight = self.mu_weight + (sigma_weight * self.eps_weight.normal_())
        kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                self.prior_weight_mu, self.prior_variance)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.normal_())
            kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                  self.prior_variance)

        out = F.linear(input, weight, bias)
        if self.mu_bias is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl

class Conv2dVariational(Module):
    def __init__(self,
                 prior_mean,
                 prior_variance,
                 posterior_mu_init,
                 posterior_rho_init,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(Conv2dVariational, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        self.posterior_rho_init = posterior_rho_init,  # variance of weight --> sigma = log (1 + exp(rho))
        self.bias = bias

        self.mu_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.rho_kernel = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        if self.bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None)
            self.register_buffer('prior_bias_mu', None)
            self.register_buffer('prior_bias_sigma', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(std=0.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = torch.log(sigma_p + 1e-15) - torch.log(
            sigma_q + 1e-15) + (sigma_q**2 +
                                (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
        return kl.sum()

    def forward(self, input):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.normal_()
        weight = self.mu_kernel + (sigma_weight * eps_kernel)
        kl_weight = self.kl_div(self.mu_kernel, sigma_weight,
                                self.prior_weight_mu, self.prior_weight_sigma)
        bias = None

        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.normal_()
            bias = self.mu_bias + (sigma_bias * eps_bias)
            kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                  self.prior_bias_sigma)

        out = F.conv2d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)
        if self.bias:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       in_planes,
                                       planes,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       planes,
                                       planes,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    Conv2dVariational(prior_mu,
                                      prior_sigma,
                                      posterior_mu_init,
                                      posterior_rho_init,
                                      in_planes,
                                      self.expansion * planes,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        out, kl = self.conv2(out)
        kl_sum += kl
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, kl_sum


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dVariational(prior_mu,
                                       prior_sigma,
                                       posterior_mu_init,
                                       posterior_rho_init,
                                       3,
                                       16,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = LinearVariational(prior_mu, prior_sigma,
                                        posterior_mu_init, posterior_rho_init,
                                        64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl
        out = self.bn1(out)
        out = F.relu(out)
        for l in self.layer1:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer2:
            out, kl = l(out)
        kl_sum += kl
        for l in self.layer3:
            out, kl = l(out)
        kl_sum += kl

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out, kl = self.linear(out)
        kl_sum += kl
        return out, kl_sum

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



