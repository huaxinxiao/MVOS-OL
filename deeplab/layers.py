import numpy as np

import torch
from torch.nn import functional as F

'''
Functional definitions of common layers
Useful for when weights are exposed rather 
than being contained in modules
'''

def linear(input, weight, bias=None):
    if bias is None:
        return F.linear(input, weight.cuda())
    else:
        return F.linear(input, weight.cuda(), bias.cuda())

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):
    if bias is None:
        return F.conv2d(input, weight.cuda(), stride=stride, padding=padding, dilation=dilation)
    else:
        return F.conv2d(input, weight.cuda(), bias.cuda(), stride=stride, padding=padding, dilation=dilation)

def batch_norm(input, running_mean, running_var, weight=None, bias=None):
    return F.batch_norm(input, running_mean.cuda(), running_var.cuda(), weight=weight.cuda(), bias=bias.cuda())

def relu(input):
    return F.threshold(input, 0, 0, inplace=True)

def maxpool(input, kernel_size, stride=1, padding=0):
    return F.max_pool2d(input, kernel_size, stride=stride, padding=padding)

def dropout(input, training=True):
    return F.dropout(input, training=training)

def log_softmax(input):
    return F.log_softmax(input)
