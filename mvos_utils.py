import cv2
import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable

img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_bn_params(model_params, train_all_features, bn_names):
    bn_params = OrderedDict()
    for i in model_params:
        if any([i.startswith(l) for l in train_all_features]):
            if 'bn' in i or any([k in i for k in bn_names]):
                if 'running' in i:
                    bn_params[i] = model_params[i]
                else:
                    bn_params[i] = Variable(model_params[i].clone(), requires_grad=False)
    return bn_params


def array2tensor(sample):
    if sample.ndim == 2:
        sample = sample[np.newaxis, np.newaxis, :]
    elif sample.ndim == 3:
        sample = sample[np.newaxis, :]
    return torch.from_numpy(sample.copy()).float()


def load_image_label_davis16(img_path, label_path=None):
    # load image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = np.asarray(image, np.float32)
    image -= img_mean
    image = image.transpose((2, 0, 1))

    # load label
    if not label_path == None:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.asarray(label, np.float32) / 255.
        return array2tensor(image), array2tensor(label)
    else:
        return array2tensor(image)


def load_image_label_davis17(img_path, crop_size, label_path=None):
    # load image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = np.asarray(image, np.float32)
    image_size = image.shape[:2]
    if not (image_size == crop_size):
        image = cv2.resize(image, crop_size[::-1])
    image -= img_mean
    image = image.transpose((2, 0, 1))

    # load label
    if not label_path == None:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.asarray(label, np.float32)
        if not (image_size == crop_size):
            label = cv2.resize(label, crop_size[::-1], interpolation = cv2.INTER_NEAREST)
        return array2tensor(image), image_size, array2tensor(label)
    else:
        return array2tensor(image), image_size


def get_trainable_params(model):
    for i, p in model.named_parameters():
        if p.requires_grad:
            yield p


def set_trainable_feas(train_mode):
    if train_mode == 'block_3_4_5':
        train_all_features = ['layer3', 'layer4', 'layer5']
    elif train_mode == 'block_4_5':
        train_all_features = ['layer4', 'layer5']
    elif train_mode == 'block_5':
        train_all_features = ['layer5']
    else:
        raise ValueError('Undefined train mode.')
    return train_all_features
