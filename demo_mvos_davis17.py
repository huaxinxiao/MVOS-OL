import sys
import os
import cv2
import os.path as osp
import numpy as np
import time
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

# from deeplab.model_dplv3_plus_meta_v3 import Res_Dplv3_Decoder 
from deeplab.model_dplv3_plus_meta_v2 import Res_Dplv3_Decoder 
from deeplab.datasets import DavisTestDataset
# from new_loss import class_balanced_cross_entropy_loss as balanced_bce_loss 

from mvos_utils import *
import pdb

# gpu setting 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.enabled = True
cudnn.benchmark = True

# meta learning setting
layer5_bn_name = ['layer5.assp.0.1', 'layer5.assp.1.1', 'layer5.assp.2.1', 'layer5.assp.3.1', 
                  'layer5.image_pool.2', 'layer5.encode_fea.1', 'layer5.low_level_fea.1',
                  'layer5.decode_1.1', 'layer5.decode_2.1']
train_mode = 'block_4_5'
meta_restore_from = './snapshots/meta_davis17_v2_2018-05-16-10:25_12000.pth'
normal_restore_from = './snapshots/deeplabv3+decoder_bnwd_0.01_lr_0.001_2018-04-16-15:51_20000.pth'
n_meta_init = 50
name_suffix = ''


# dataset setting
cfg = {}
cfg['img_root'] = './dataset/davis/JPEGImages/480p_split'
cfg['gt_root'] = './dataset/davis/Annotations/480p_split'
cfg['list_root'] = './dataset/davis/ImageSets/2017/val_w_imgs_no.txt'
cfg['crop_size'] = (480, 854)
cfg['img_mean'] = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
cfg['random_scale'] = False
cfg['random_mirror'] = False
cfg['ignore_label'] = 0
cfg['num_steps'] = n_meta_init
cfg['batch_size'] = 1
cfg['max_label'] = 1.0


# get sequences
img_ids = [i_id.strip() for i_id in open(cfg['list_root'])]
seq_numbers = len(img_ids)


# create network
model = Res_Dplv3_Decoder(num_classes=1, output_stride=16)
interp = nn.Upsample(size=cfg['crop_size'], mode='bilinear')
criterion = nn.BCEWithLogitsLoss().cuda()
sigmoid = nn.Sigmoid()


# testing save root
meta_model_name = meta_restore_from.split('/')[-1][:-4]
save_root_prefix = '_'.join(['./results/davis-17-val/' + meta_model_name, str(n_meta_init), name_suffix])

total_runtime = 0.
# process sequence
for i in range(seq_numbers):
    cur_seq_name = img_ids[i].split()[0]
    cur_seq_range = int(img_ids[i].split()[1])
    cur_seq_image_path = osp.join(cfg['img_root'], cur_seq_name)
    cur_seq_label_path = osp.join(cfg['gt_root'], cur_seq_name)

    # set learnable parameters
    train_all_features = set_trainable_feas(train_mode)
    model.set_learnable_params(train_all_features, layer5_bn_name)

    # load non-meta-trained model
    saved_state_dict = torch.load(normal_restore_from)
    model.load_state_dict(saved_state_dict)
    bn_params = get_bn_params(saved_state_dict, train_all_features, layer5_bn_name)

    # load meta-trained model
    saved_state_dict = torch.load(meta_restore_from)
    meta_init = saved_state_dict['meta_init']
    meta_alpha = saved_state_dict['meta_alpha']
    model.copy_meta_weights(meta_init)
    model.train()
    model.cuda()

    # meta-initialize the model
    model_weights = OrderedDict()
    for name, p in model.named_parameters():
        if name.split('.')[0] in train_all_features and name not in bn_params:
            model_weights[name] = p

    # frozen bn or not
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    ol_trainloader = data.DataLoader(DavisTestDataset(cfg, cur_seq_name), batch_size=cfg['batch_size'],
                                     shuffle=True, num_workers=0, pin_memory=True)

    # meta-finetune on first frame
    tic = time.time()
    print ("Meta-Training on %s..." % cur_seq_name)
    for i_iter, batch in enumerate(ol_trainloader):
        image, label = batch
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        pred = interp(model.forward(image, model_weights, train_mode, bn_params))

        loss = criterion(pred, label)
        grads = torch.autograd.grad(loss, model_weights.values(), retain_graph=False, create_graph=True)
        model_weights = OrderedDict((name, param - torch.mul(meta_alpha,grad)) for 
                                    ((name, param), (_, meta_alpha), grad) in
                                    zip(model_weights.items(), meta_alpha.items(), grads))
        #print ('loss is %f' % loss.data.cpu().numpy())
        
        #if i_iter == n_meta_init:
        #    break

    print ('Meta fine-tuning time is : %.2f' % (time.time() - tic))

    # update model parameters
    model.copy_meta_weights(model_weights)


    # online testing
    model.eval()
    print ("Evaluate on %s..." % cur_seq_name)
    cur_save_root = save_root_prefix + '/' + cur_seq_name + '/'
    if not osp.exists(cur_save_root):
        os.makedirs(cur_save_root)

    output = label.data[0, 0].cpu().numpy()
    output = np.array(output*255, dtype=np.uint8)
    cv2.imwrite(cur_save_root + '00000.png', output)
    for j in range(1, cur_seq_range):
        # tic = time.time()
        img_file = osp.join(cur_seq_image_path, '%05d' % j + '.jpg')
        img, ori_img_size = load_image_label_davis17(img_file, cfg['crop_size'])
        output = model(Variable(img, volatile = True).cuda())
        output = sigmoid(interp(output)).cpu().data[0,0].numpy()
        # print ('Per frame forward time is : %.2f' % (time.time() - tic))
        if not (ori_img_size==cfg['crop_size']):
            #output = cv2.resize(output, ori_img_size[::-1], interpolation = cv2.INTER_NEAREST)
            output = cv2.resize(output, ori_img_size[::-1], interpolation = cv2.INTER_LINEAR)
        output = np.array(output*255, dtype=np.uint8)
        cv2.imwrite(cur_save_root + '/%05d' % j + '.png', output)

    total_runtime += (time.time() - tic) / (cur_seq_range - 1)

print ('Average per-frame runtime is : {}'.format(total_runtime/seq_numbers))
