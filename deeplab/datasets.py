import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import pdb


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class VOCSalDataSet_Old(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy(), np.array(size), name

class VOCSalDataSet(data.Dataset):
    def __init__(self, cfg):
        self.root = cfg['img_root']
        self.list_root = cfg['list_root']
        self.crop_h, self.crop_w = cfg['input_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']

        # get list
        max_iters = cfg['num_steps'] * cfg['batch_size']
        self.img_ids = [i_id.strip() for i_id in open(self.list_root)]
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            # img_file = self.root + name.split()[0]
            # label_file = self.root + name.split()[1]
            self.files.append({
                "img": img_file,
                "label": label_file,
            })
        self.files = self.files * int(np.ceil(float(max_iters) / len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # size = image.shape
        # name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy()

class DavisTrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.img_root = cfg['img_root']
        self.gt_root = cfg['gt_root']
        self.list_root = cfg['list_root']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']

        #get list
        max_iters = cfg['num_steps'] * cfg['batch_size']
        img_ids = [i_id.strip() for i_id in open(self.list_root)]
        self.seq_numbers = len(img_ids)
        self.files = []
        for i in range(self.seq_numbers):
            cur_seq_name = img_ids[i].split()[0]
            for j in range(int(img_ids[i].split()[1])):
                name = '%s/%05d'%(cur_seq_name, j)
                img_file = os.path.join(self.img_root, name + '.jpg')
                gt_file = os.path.join(self.gt_root, name + '.png')
                self.files.append({
                    "img": img_file,
                    "label": gt_file,
                    "name": name
                })
        self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))


    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy()


class DavisTestDataset(data.Dataset):
    def __init__(self, cfg, seq_name):
        self.img_root = cfg['img_root']
        self.gt_root = cfg['gt_root']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']
        self.max_label = cfg['max_label']

        #get list
        max_iters = cfg['num_steps'] * cfg['batch_size']
        self.files = []
        name = '%s/%05d'%(seq_name, 0)
        img_file = os.path.join(self.img_root, name + '.jpg')
        gt_file = os.path.join(self.gt_root, name + '.png')
        self.files.append({
            "img": img_file,
            "label": gt_file,
            "name": name
        })
        self.files = self.files * max_iters

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / self.max_label
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy()

class YoutubeTestDataset(data.Dataset):
    def __init__(self, cfg, seq_name):
        self.img_root = cfg['img_root']
        self.gt_root = cfg['gt_root']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']
        self.max_label = cfg['max_label']

        #get list
        max_iters = cfg['num_steps'] * cfg['batch_size']
        self.files = []
        name = '%s/%05d'%(seq_name, 1)
        img_file = os.path.join(self.img_root, name + '.png')
        gt_file = os.path.join(self.gt_root, name + '.png')
        self.files.append({
            "img": img_file,
            "label": gt_file,
            "name": name
        })
        self.files = self.files * max_iters

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / self.max_label
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy()


class DavisTestDataset_v7(data.Dataset):
    def __init__(self, cfg, seq_name):
        self.img_root = cfg['img_root']
        self.gt_root = cfg['gt_root']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = True
        self.is_mirror = True
        self.ignore_label = 0

        #get list
        max_iters = cfg['num_steps'] * cfg['batch_size']
        self.files = []
        name = '%s/%05d'%(seq_name, 0)
        img_file = os.path.join(self.img_root, name + '.jpg')
        gt_file = os.path.join(self.gt_root, name + '.png')
        self.files.append({
            "img": img_file,
            "label": gt_file,
            "name": name
        })
        self.files = self.files * max_iters

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label[np.newaxis, :]
        return image.copy(), label.copy()


class DavisMetaTrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.dataset_root = cfg['dataset_root']
        self.list_root = cfg['davis_list']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']

        #get list
        img_ids = [i_id.strip() for i_id in open(self.list_root)]
        self.seq_numbers = len(img_ids)
        self.image_names = []
        self.label_names = []
        self.seq_ranges = []
        for i in range(self.seq_numbers):
            self.image_names.append(img_ids[i].split()[0])
            self.label_names.append(img_ids[i].split()[1])
            self.seq_ranges.append(int(img_ids[i].split()[2]))

        self.lookahead = cfg['lookahead']

    def __iter__(self):
        return self

    def generate_scale_label(self, image, label, lh_image, lh_label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        lh_image = cv2.resize(lh_image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        lh_label = cv2.resize(lh_label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, lh_image, lh_label
    
    def array2tensor(self, sample):
        if sample.ndim == 2:
            sample = sample[np.newaxis, np.newaxis, :]
        elif sample.ndim == 3:
            sample = sample[np.newaxis, :]
        return torch.from_numpy(sample.copy()).float()

    def __next__(self):
        seq_id = np.random.randint(self.seq_numbers)
        idx = np.random.randint(self.seq_ranges[seq_id] - self.lookahead)
        lh_idx = idx + np.random.randint(self.lookahead) + 1

        img_path = self.dataset_root + ('/%s/%05d.jpg'%(self.image_names[seq_id], idx))
        gt_path = self.dataset_root + ('/%s/%05d.png'%(self.label_names[seq_id], idx))
        lh_img_path = self.dataset_root + ('/%s/%05d.jpg'%(self.image_names[seq_id], lh_idx))
        lh_gt_path = self.dataset_root + ('/%s/%05d.png'%(self.label_names[seq_id], lh_idx))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lh_img = cv2.imread(lh_img_path, cv2.IMREAD_COLOR)
        lh_gt = cv2.imread(lh_gt_path, cv2.IMREAD_GRAYSCALE)

        # random scale 
        if self.is_scale:
            img, gt, lh_img, lh_gt = self.generate_scale_label(img, gt, lh_img, lh_gt)

        # padding 
        img = np.asarray(img, np.float32)
        lh_img = np.asarray(lh_img, np.float32)
        img -= self.mean
        lh_img -= self.mean

        img_h, img_w = gt.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            gt_pad = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
            lh_img_pad = cv2.copyMakeBorder(lh_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            lh_gt_pad = cv2.copyMakeBorder(lh_gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
        else:
            img_pad, gt_pad, lh_img_pad, lh_gt_pad = img, gt, lh_img, lh_gt

        # random crop
        img_h, img_w = gt_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        img = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        gt = np.asarray(gt_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.
        lh_img = np.asarray(lh_img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        lh_gt = np.asarray(lh_gt_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.

        img = img.transpose((2, 0, 1))
        lh_img = lh_img.transpose((2, 0, 1))

        # random mirror
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            gt = gt[:, ::flip]
            lh_img = lh_img[:, :, ::flip]
            lh_gt = lh_gt[:, ::flip]

        return self.array2tensor(img), self.array2tensor(gt), self.array2tensor(lh_img), self.array2tensor(lh_gt)

    next = __next__

class SalVideoMetaTrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.dataset_root = cfg['dataset_root']
        self.list_root = cfg['sal_video_list']
        self.crop_h, self.crop_w = cfg['crop_size']
        self.mean = cfg['img_mean']
        self.is_scale = cfg['random_scale']
        self.is_mirror = cfg['random_mirror']
        self.ignore_label = cfg['ignore_label']

        #get list
        img_ids = [i_id.strip() for i_id in open(self.list_root)]
        self.seq_numbers = len(img_ids)
        self.image_names = []
        self.label_names = []
        self.seq_ranges = []
        for i in range(self.seq_numbers):
            self.image_names.append(img_ids[i].split()[0])
            self.label_names.append(img_ids[i].split()[1])
            self.seq_ranges.append(int(img_ids[i].split()[2]))

        self.lookahead = 1

    def __iter__(self):
        return self

    def generate_scale_label(self, image, label, lh_image, lh_label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        lh_image = cv2.resize(lh_image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        lh_label = cv2.resize(lh_label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, lh_image, lh_label
    
    def array2tensor(self, sample):
        if sample.ndim == 2:
            sample = sample[np.newaxis, np.newaxis, :]
        elif sample.ndim == 3:
            sample = sample[np.newaxis, :]
        return torch.from_numpy(sample.copy()).float()

    def __next__(self):
        seq_id = np.random.randint(self.seq_numbers)
        idx = np.random.randint(self.seq_ranges[seq_id] - self.lookahead)
        lh_idx = idx + 1

        img_path = self.dataset_root + ('/%s/%05d.jpg'%(self.image_names[seq_id], idx))
        gt_path = self.dataset_root + ('/%s/%05d.png'%(self.label_names[seq_id], idx))
        lh_img_path = self.dataset_root + ('/%s/%05d.jpg'%(self.image_names[seq_id], lh_idx))
        lh_gt_path = self.dataset_root + ('/%s/%05d.png'%(self.label_names[seq_id], lh_idx))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lh_img = cv2.imread(lh_img_path, cv2.IMREAD_COLOR)
        lh_gt = cv2.imread(lh_gt_path, cv2.IMREAD_GRAYSCALE)

        # random scale 
        if self.is_scale:
            img, gt, lh_img, lh_gt = self.generate_scale_label(img, gt, lh_img, lh_gt)

        # padding 
        img = np.asarray(img, np.float32)
        lh_img = np.asarray(lh_img, np.float32)
        img -= self.mean
        lh_img -= self.mean

        img_h, img_w = gt.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            gt_pad = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
            lh_img_pad = cv2.copyMakeBorder(lh_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            lh_gt_pad = cv2.copyMakeBorder(lh_gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,))
        else:
            img_pad, gt_pad, lh_img_pad, lh_gt_pad = img, gt, lh_img, lh_gt

        # random crop
        img_h, img_w = gt_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        img = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        gt = np.asarray(gt_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.
        lh_img = np.asarray(lh_img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        lh_gt = np.asarray(lh_gt_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32) / 255.

        img = img.transpose((2, 0, 1))
        lh_img = lh_img.transpose((2, 0, 1))

        # random mirror
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            gt = gt[:, ::flip]
            lh_img = lh_img[:, :, ::flip]
            lh_gt = lh_gt[:, ::flip]

        return self.array2tensor(img), self.array2tensor(gt), self.array2tensor(lh_img), self.array2tensor(lh_gt)

    next = __next__


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
