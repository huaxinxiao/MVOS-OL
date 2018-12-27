import torch.nn as nn
import torch
import numpy as np
from deeplab.layers import *
from collections import OrderedDict

affine_par = True

def fun_bottleneck_features(x, weights, bn_params, layer_name, dilation, residual):
    out = conv2d(x, weights[layer_name + 'conv1.weight'], stride=1)
    out = batch_norm(out, bn_params[layer_name + 'bn1.running_mean'], bn_params[layer_name + 'bn1.running_var'],
                     weights[layer_name + 'bn1.weight'], weights[layer_name + 'bn1.bias'])
    out = relu(out)

    out = conv2d(out, weights[layer_name + 'conv2.weight'], stride=1, padding=dilation, dilation=dilation)
    out = batch_norm(out, bn_params[layer_name + 'bn2.running_mean'], bn_params[layer_name + 'bn2.running_var'],
                     weights[layer_name + 'bn2.weight'], weights[layer_name + 'bn2.bias'])
    out = relu(out)

    out = conv2d(out, weights[layer_name + 'conv3.weight'], stride=1)
    out = batch_norm(out, bn_params[layer_name + 'bn3.running_mean'], bn_params[layer_name + 'bn3.running_var'],
                     weights[layer_name + 'bn3.weight'], weights[layer_name + 'bn3.bias'])

    out += residual
    out = relu(out)
    return out

def fun_block_features(x, layer_cfg, weights, bn_params):
    if layer_cfg == 23:
        dilation = 2
        layer_name = 'layer3.'
    elif layer_cfg == 3:
        dilation = 4
        layer_name = 'layer4.'
    else:
        raise ValueError('Undefined layer configuration')

    # first layer with downsample
    bn_name = layer_name + '0.downsample.'
    downsample = conv2d(x, weights[bn_name + 'conv.weight'])
    downsample = batch_norm(downsample, bn_params[bn_name + 'bn.running_mean'], bn_params[bn_name + 'bn.running_var'],
                            weights[bn_name + 'bn.weight'], weights[bn_name + 'bn.bias'])
    out = fun_bottleneck_features(x, weights, bn_params, layer_name+'0.', dilation, downsample)

    for i in range(1, layer_cfg):
        residual = out
        cur_layer_name = layer_name + str(i) + '.'
        out = fun_bottleneck_features(out, weights, bn_params, cur_layer_name, dilation, residual)

    return out

def fun_assp_features(x, weights):
     x0 = conv2d(x, weights['layer5.assp0.weight'], weights['layer5.assp0.bias'], stride=1, padding=6,  dilation=6)
     x1 = conv2d(x, weights['layer5.assp1.weight'], weights['layer5.assp1.bias'], stride=1, padding=12, dilation=12)
     x2 = conv2d(x, weights['layer5.assp2.weight'], weights['layer5.assp2.bias'], stride=1, padding=18, dilation=18)
     x3 = conv2d(x, weights['layer5.assp3.weight'], weights['layer5.assp3.bias'], stride=1, padding=24, dilation=24)
     return x0 + x1 + x2 + x3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.assp0 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=dilation_series[0], 
                               dilation=dilation_series[0], bias = True)
        self.assp1 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=dilation_series[1], 
                               dilation=dilation_series[1], bias = True)
        self.assp2 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=dilation_series[2], 
                               dilation=dilation_series[2], bias = True)
        self.assp3 = nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=dilation_series[3], 
                               dilation=dilation_series[3], bias = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.assp0(x) + self.assp1(x) + self.assp2(x) + self.assp3(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(planes * block.expansion, affine = affine_par))]))

        for i in downsample._modules['bn'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, num_classes):
        return block(dilation_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class ResNet_Meta(nn.Module):
    def __init__(self, block, layer_cfg, num_classes):
        self.inplanes = 64
        self.layer_cfg = layer_cfg
        super(ResNet_Meta, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block,  64, self.layer_cfg[0])
        self.layer2 = self._make_layer(block, 128, self.layer_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, self.layer_cfg[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, self.layer_cfg[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(planes * block.expansion, affine = affine_par))]))

        for i in downsample._modules['bn'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, num_classes):
        return block(dilation_series, num_classes)

    def forward(self, x, weights=None, bn_params=None, train_mode=None):
        if weights == None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            if train_mode == 'block_3_4_5':
                x = fun_block_features(x, self.layer_cfg[2], weights, bn_params)
                x = fun_block_features(x, self.layer_cfg[3], weights, bn_params)
                x = fun_assp_features(x, weights)
            elif train_mode == 'block_4_5':
                x = self.layer3(x)
                x = fun_block_features(x, self.layer_cfg[3], weights, bn_params)
                x = fun_assp_features(x, weights)
            elif train_mode == 'block_5':
                x = self.layer3(x)
                x = self.layer4(x)
                x = fun_assp_features(x, weights)
        return x

    def set_learnable_params(self, layers):
        print ('Set meta-learning parameters:', end=' ')
        for k, p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
                print (k, end=' ')
            else:
                p.requires_grad = False
        print ('\n')

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

def Res_Deeplab(num_classes=21, meta_train=False):
    if meta_train:
        model = ResNet_Meta(Bottleneck, [3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

