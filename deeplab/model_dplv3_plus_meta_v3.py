import torch.nn as nn
import torch
import numpy as np
from deeplab.layers import *
from torch.nn import functional as F
from collections import OrderedDict

affine_par = True

def fun_bottleneck_fea(x, weights, bn_params, layer_name, dilation, residual, stride):
    out = conv2d(x, weights[layer_name + 'conv1.weight'], stride=stride)
    out = batch_norm(out, bn_params[layer_name + 'bn1.running_mean'], bn_params[layer_name + 'bn1.running_var'],
                     bn_params[layer_name + 'bn1.weight'], bn_params[layer_name + 'bn1.bias'])
    out = relu(out)

    out = conv2d(out, weights[layer_name + 'conv2.weight'], stride=1, padding=dilation, dilation=dilation)
    out = batch_norm(out, bn_params[layer_name + 'bn2.running_mean'], bn_params[layer_name + 'bn2.running_var'],
                     bn_params[layer_name + 'bn2.weight'], bn_params[layer_name + 'bn2.bias'])
    out = relu(out)

    out = conv2d(out, weights[layer_name + 'conv3.weight'], stride=1)
    out = batch_norm(out, bn_params[layer_name + 'bn3.running_mean'], bn_params[layer_name + 'bn3.running_var'],
                     bn_params[layer_name + 'bn3.weight'], bn_params[layer_name + 'bn3.bias'])

    out += residual
    out = relu(out)
    return out


def fun_residual_fea(x, weights, bn_params, layer_cfg):

    if layer_cfg == 'layer3':
        stride = 2
        dilation = [1]*23
        layer_name = layer_cfg + '.'
    elif layer_cfg == 'layer4':
        stride = 1
        dilation = [2, 4, 8]
        layer_name = layer_cfg + '.'
    else:
        raise ValueError('Undefined layer_cfg: {}!'.format(layer_cfg))

    # first layer with downsample
    bn_name = layer_name + '0.downsample.'
    downsample = conv2d(x, weights[bn_name + 'conv.weight'], stride=stride)
    downsample = batch_norm(downsample, bn_params[bn_name + 'bn.running_mean'], bn_params[bn_name + 'bn.running_var'],
                            bn_params[bn_name + 'bn.weight'], bn_params[bn_name + 'bn.bias'])

    out = fun_bottleneck_fea(x, weights, bn_params, layer_name+'0.', dilation[0], downsample, stride)

    for i in range(1, len(dilation)):
        residual = out
        cur_layer_name = layer_name + str(i) + '.'
        out = fun_bottleneck_fea(out, weights, bn_params, cur_layer_name, dilation[i], residual, 1)

    return out

def fub_cbr_fea(x, weights, bn_params, conv_name, bn_name, padding, dilation):
    out = conv2d(x, weights[conv_name+'.weight'], padding=padding, dilation=dilation)
    out = batch_norm(out, bn_params[bn_name + '.running_mean'], bn_params[bn_name + '.running_var'],
                    bn_params[bn_name + '.weight'], bn_params[bn_name + '.bias'])
    out = relu(out)
    return out

def fun_layer5_fea(res2_fea, res5_fea, weights, bn_params):
     x0 = fub_cbr_fea(res5_fea, weights, bn_params, 'layer5.assp.0.0', 'layer5.assp.0.1', 0, 1)
     x1 = fub_cbr_fea(res5_fea, weights, bn_params, 'layer5.assp.1.0', 'layer5.assp.1.1', 6, 6)
     x2 = fub_cbr_fea(res5_fea, weights, bn_params, 'layer5.assp.2.0', 'layer5.assp.2.1', 12, 12)
     x3 = fub_cbr_fea(res5_fea, weights, bn_params, 'layer5.assp.3.0', 'layer5.assp.3.1', 18, 18)

     g_pool = F.adaptive_avg_pool2d(res5_fea, output_size=(1, 1))
     x4 = fub_cbr_fea(g_pool, weights, bn_params, 'layer5.image_pool.1', 'layer5.image_pool.2', 0, 1)
     x4 = F.upsample(input=x4, size=(res5_fea.size(2), res5_fea.size(3)), mode='bilinear')
     encoder_fea = torch.cat((x0, x1, x2, x3, x4), dim=1)
     encoder_fea = fub_cbr_fea(encoder_fea, weights, bn_params, 'layer5.encode_fea.0', 'layer5.encode_fea.1', 0, 1)

     # get low level feature
     low_level_fea = fub_cbr_fea(res2_fea, weights, bn_params, 'layer5.low_level_fea.0', 'layer5.low_level_fea.1', 0, 1)
     encoder_fea = F.upsample(input=encoder_fea, size=(res2_fea.size(2), res2_fea.size(3)), mode='bilinear')
     out = torch.cat((encoder_fea, low_level_fea), dim=1)
     out = fub_cbr_fea(out, weights, bn_params, 'layer5.decode_1.0', 'layer5.decode_1.1', 1, 1)
     out = fub_cbr_fea(out, weights, bn_params, 'layer5.decode_2.0', 'layer5.decode_2.1', 1, 1)
     out = conv2d(out, weights['layer5.final.weight'], weights['layer5.final.bias'], stride=1, padding=0, dilation=1)
     return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4, affine = affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

class Deeplabv3PlusDecoderHead(nn.Module):

    def __init__(self, dilations, num_classes):
        super(Deeplabv3PlusDecoderHead, self).__init__()

        self.assp = []
        self.assp = nn.ModuleList([self._make_assp(2048, 256, dilation) for dilation in dilations])
        self.image_pool = self._make_image_fea(2048, 256)

        self.encode_fea = self._make_block(256*5, 256, ks=1, pad=0)
        self.low_level_fea = self._make_block(256, 48, ks=1, pad=0)
        self.decode_1 = self._make_block(256+48, 256, ks=3, pad=1)
        self.decode_2 = self._make_block(256, 256, ks=3, pad=1)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)

    def _make_assp(self, in_features, out_features, dilation):
        if dilation == 1:
            conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        else:
            conv = nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def _make_image_fea(self, in_features, out_features):
        g_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(g_pool, conv, bn, relu)

    def _make_block(self, in_features, out_features, ks=1, pad=0):
        conv = nn.Conv2d(in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, res2_fea, res5_fea):
        # get encoder feature
        h, w = res5_fea.size(2), res5_fea.size(3)
        image_fea = F.upsample(input=self.image_pool(res5_fea), size=(h, w), mode='bilinear')
        encoder_fea = [assp_stage(res5_fea) for assp_stage in self.assp]
        encoder_fea.append(image_fea)
        encoder_fea = self.encode_fea(torch.cat(encoder_fea, dim=1))

        # get low level feature
        low_level_fea = self.low_level_fea(res2_fea)

        # concat and decode
        h, w = res2_fea.size(2), res2_fea.size(3)
        encoder_fea = F.upsample(input=encoder_fea, size=(h, w), mode='bilinear')
        out = self.decode_1(torch.cat((encoder_fea, low_level_fea), dim=1))
        out = self.decode_2(out)
        out = self.final(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, output_stride):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # setting dilated convolution rates
        if output_stride == 16:
            layer3_stride, layer4_stride = 2, 1
            layer3_dilation = [1]*layers[2]
            layer4_dilation = [2, 4, 8]
        elif output_stride == 8:
            layer3_stride, layer4_stride = 1, 1
            layer3_dilation = [2]*layers[2]
            layer4_dilation = [4, 8, 16]
        else:
            raise ValueError('Undifined output stride rate!')

        # build network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, dilation=[1]*layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=[1]*layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=layer3_stride, dilation=layer3_dilation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer4_stride, dilation=layer4_dilation)
        self.layer5 = Deeplabv3PlusDecoderHead((1,6,12,18), num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, dilation):
        downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(planes * block.expansion, affine = affine_par))]))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation[i]))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, num_classes):
        return block(dilation_series, num_classes)

    def forward(self, x, weights=None, train_mode=None, bn_state=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        res2_fea = self.layer1(x)
        x = self.layer2(res2_fea)

        if train_mode == 'block_3_4_5':
            x = fun_residual_fea(x, weights, bn_state, 'layer3')
            x = fun_residual_fea(x, weights, bn_state, 'layer4')
            x = fun_layer5_fea(res2_fea, x, weights, bn_state)
        elif train_mode == 'block_4_5':
            x = self.layer3(x)
            x = fun_residual_fea(x, weights, bn_state, 'layer4')
            x = fun_layer5_fea(res2_fea, x, weights, bn_state)
        elif train_mode == 'block_5':
            x = self.layer3(x)
            x = self.layer4(x)
            x = fun_layer5_fea(res2_fea, x, weights, bn_state)
        elif train_mode == None:
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(res2_fea, x)
        else:
            raise ValueError('Undefined training mode: {}.'.format(train_mode))

        return x

    def set_learnable_params(self, layers, bn_names):
        print ('Set meta learn parameters:', end=' ')
        for name, p in self.named_parameters():
            if any([name.startswith(l) for l in layers]):
                if 'bn' in name or any([k in name for k in bn_names]):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    print (name, end='-->')
            else:
                p.requires_grad = False
        print ('END')

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()


def Res_Dplv3_Decoder(num_classes, output_stride):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, output_stride)

