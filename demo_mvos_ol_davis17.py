import sys
import os
import cv2
import os.path as osp
import numpy as np
import time
import pdb
from collections import OrderedDict
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

from deeplab.model_dplv3_plus_meta_v2 import Res_Dplv3_Decoder 
from deeplab.datasets import DavisTestDataset
from new_loss import class_cross_entropy_loss as my_bce_loss 

from mvos_utils import *

# gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.enabled = True
cudnn.benchmark = True


# meta learning setting
layer5_bn_name = ['layer5.assp.0.1', 'layer5.assp.1.1', 'layer5.assp.2.1', 'layer5.assp.3.1', 
                  'layer5.image_pool.2', 'layer5.encode_fea.1', 'layer5.low_level_fea.1',
                  'layer5.decode_1.1', 'layer5.decode_2.1']
meta_restore_from = './snapshots/meta_davis17.pth'
normal_restore_from = './snapshots/base_seg.pth'
n_meta_init = 50
n_meta_update = 2
train_mode = 'block_4_5'


# dataset setting
cfg = {}
cfg['img_root'] = './dataset/davis/JPEGImages/480p'
cfg['gt_root'] = './dataset/davis/Annotations/480p_split'
cfg['list_root'] = './dataset/davis/ImageSets/2017/val_w_imgs_no.txt'
cfg['crop_size'] = (480, 854)
cfg['img_mean'] = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
cfg['random_scale'] = True
cfg['random_mirror'] = True
cfg['ignore_label'] = 0
cfg['num_steps'] = n_meta_init
cfg['batch_size'] = 1
cfg['max_label'] = 1.0


# online adaptation parameters
pos_pred_thres = 0.90
neg_dist_thres = 120.0
adapt_erosion_size = 15
adapt_ignore_label = 255
adapt_debug = True
adapt_interval = 2

beta_0 = 0.99
beta_1 = 1.0 - beta_0


# get sequences
img_ids = [i_id.strip() for i_id in open(cfg['list_root'])]
seq_numbers = len(img_ids)


# create network
model = Res_Dplv3_Decoder(num_classes=1, output_stride=16)
interp = nn.Upsample(size=cfg['crop_size'], mode='bilinear')
# criterion = nn.BCEWithLogitsLoss().cuda()
sigmoid = nn.Sigmoid()


# testing save root
name_suffix = 'adapt_{}_{}_{}_{}_{}'.format(n_meta_init, n_meta_update, adapt_interval, pos_pred_thres, neg_dist_thres) + \
                '_beta_{:.2f}_{:.2f}'.format(beta_0, beta_1)
meta_model_name = meta_restore_from.split('/')[-1][:-4]
save_root_prefix = '_'.join(['./results/davis-17-val/' + meta_model_name, name_suffix])

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

    # meta-initialize on first frame
    tic = time.time()
    print ("Meta-Training on %s..." % cur_seq_name)
    for i_iter, batch in enumerate(ol_trainloader):
        first_image, first_label = batch
        first_image = Variable(first_image).cuda()
        first_label = Variable(first_label).cuda()

        pred = interp(model.forward(first_image, model_weights, train_mode, bn_params))

        loss = my_bce_loss(pred, first_label)
        grads = torch.autograd.grad(loss, model_weights.values(), retain_graph=False, create_graph=True)
        model_weights = OrderedDict((name, param - torch.mul(meta_alpha,grad)) for 
                                    ((name, param), (_, meta_alpha), grad) in
                                    zip(model_weights.items(), meta_alpha.items(), grads))
        
        # if i_iter == n_meta_init:
        #     break

    print ('Meta-initialize time is : %.2f' % (time.time() - tic))


    # online adaptation and testing
    print ("Online adaptation on %s..." % cur_seq_name)
    cur_save_root = save_root_prefix + '/' + cur_seq_name + '/'
    if not osp.exists(cur_save_root):
        os.makedirs(cur_save_root)

    last_mask = first_label.data[0, 0].cpu().numpy()
    output = np.array(last_mask*255, dtype=np.uint8)
    cv2.imwrite(cur_save_root + '00000.png', output)

    for j in range(1, cur_seq_range):
        img_file = osp.join(cur_seq_image_path, '%05d' % j + '.jpg')
        cur_img, ori_img_size = load_image_label_davis17(img_file, cfg['crop_size'])
        cur_img = Variable(cur_img).cuda()
        output = model.forward(cur_img, model_weights, train_mode, bn_params)
        output = sigmoid(interp(output)).cpu().data[0, 0].numpy()

        # use current frame to generate label for next frame
        gen_label = np.zeros_like(output)
        gen_label[:] = adapt_ignore_label

        eroded_mask = grey_erosion(last_mask, size=(adapt_erosion_size, adapt_erosion_size))
        eroded_mask[eroded_mask<0.1] = 0
        dt = distance_transform_edt(np.logical_not(eroded_mask))
        
        gen_label[output > pos_pred_thres] = 1
        gen_label[dt > neg_dist_thres] = 0

        do_adapt = eroded_mask.sum() > 0

        if adapt_debug:
            if do_adapt:
                gen_label_vis = gen_label.copy()
                gen_label_vis[gen_label==1] = 128
            else:
                gen_label_vis[:] = adapt_ignore_label
            cv2.imwrite(cur_save_root + 'adapt_%05d' % j + '.png', gen_label_vis)


        # online adaptation
        if j%adapt_interval == 0:
            for iter_ol_adapt in range(n_meta_update):

                # get first frame gradient
                pred = interp(model.forward(first_image, model_weights, train_mode, bn_params))
                loss = my_bce_loss(pred, first_label)
                first_grads = torch.autograd.grad(loss, model_weights.values(), retain_graph=False, create_graph=True)
                first_grads = list(first_grads)

                if do_adapt:
                    ol_adapt_label = Variable(array2tensor(gen_label)).cuda()
                    pred = interp(model.forward(cur_img, model_weights, train_mode, bn_params))
                    loss = my_bce_loss(pred, ol_adapt_label, adapt_ignore_label)
                    grads = torch.autograd.grad(loss, model_weights.values(), retain_graph=False, create_graph=True)

                    for iter_grad in range(len(grads)):
                        first_grads[iter_grad] = torch.mul(grads[iter_grad], beta_1) + \
                                                 torch.mul(first_grads[iter_grad], beta_0)

                # update model weigths
                first_grads = tuple(first_grads)
                model_weights = OrderedDict((name, param - torch.mul(meta_alpha,grad)) for 
                                            ((name, param), (_, meta_alpha), grad) in
                                            zip(model_weights.items(), meta_alpha.items(), first_grads))

            output = model.forward(cur_img, model_weights, train_mode, bn_params)
            output = sigmoid(interp(output)).cpu().data[0, 0].numpy()

        if do_adapt:
            output[dt > neg_dist_thres] = 0
        last_mask = output
        if not (ori_img_size==cfg['crop_size']):
            output = cv2.resize(output, ori_img_size[::-1], interpolation = cv2.INTER_LINEAR)
        output = np.array(output*255, dtype=np.uint8)
        cv2.imwrite(cur_save_root + '%05d' % j + '.png', output)
    
    total_runtime += (time.time() - tic) / (cur_seq_range - 1)
print ('Average per-frame runtime is : {}'.format(total_runtime/seq_numbers))
