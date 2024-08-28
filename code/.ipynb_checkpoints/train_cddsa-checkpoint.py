#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    :  2022-12-20 19:10
# @Author  : Ran Gu
"""
Apply feature disentanglement and novel contrastive learning in multi sites fundus segmentation
"""
import argparse
import logging
import os
import sys
import yaml
import math
import tqdm
import random
import torch
import timeit
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.losses import *
from datetime import datetime
from torchvision import transforms
from torchvision.utils import make_grid
from dataloader.breast_ultrasound.breastultra_dataloader import BreastSegmentation
from dataloader.breast_ultrasound import breastultra_transforms as tr
# from dataloader.ms_fundus.fundus_dataloader import FundusSegmentation
# from dataloader.ms_fundus import fundus_transforms as tr
from torch.utils.data import ConcatDataset, DataLoader
from models.networks.sdnet import MEncoder, AEncoder, Segmentor, Ada_Decoder
from models.weight_init import initialize_weights
from utils.average_meter import AverageMeter
from utils.utils_fundus import sample_minibatch_fundus
from tensorboardX import SummaryWriter
from pytorch_metric_learning import losses
torch.set_default_tensor_type('torch.FloatTensor')

#Global Guidance Net Module
from models.GlobalGuidance_Net.Seg_Module.modeling.deeplab_dense import *

import cv2

import copy

def parse_args():
    desc = "Pytorch implementation of CDDSA (RanGu)"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=123, help='random seed')
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./output/cddsa/breastultrasound')
    parser.add_argument('--data_dir', type=str, default='../Data/BreastUltrasound')
    # data config
    parser.add_argument('--data_size', type=int, default=256)
    # GPU config
    parser.add_argument('--gpu', type=str, default='0')
    # training config
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--datasetTrain', nargs='+', type=int, default=[1,2,3,4], help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=[5], help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--z_length', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=14)
    parser.add_argument('--kl_w', type=float, default=0.001)
    parser.add_argument('--seg_w', type=float, default=1)
    parser.add_argument('--reco_w', type=float, default=1)
    parser.add_argument('--recoz_w', type=float, default=1)
    parser.add_argument('--style_w', type=float, default=0.1)
    parser.add_argument('--cont_w', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_minibatch', type=int, default=2)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wi', '--weight_init', type=str, default="xavier",
                        help='Weight initialization method, or path to weights file '
                             '(for fine-tuning or continuing training)')
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.4)

    # utils.check_folder(parser.parse_args().exp_dir)
    return parser.parse_args()

def rectangle_creator(pred, threshold):
    pred = torch.sigmoid(pred)
    pred[pred>=threshold] = 1
    pred[pred<threshold] = 0
    pred = pred.cpu().numpy()
    
    # file_handle=open('1.txt',mode='w')#w 写入模式
    # #将numpy类型转化为list类型
    # x=pred.tolist()
    # #将list转化为string类型
    # strNums=[str(x_i) for x_i in x]
    # str1=",".join(strNums)
    # #将str类型数据存入本地文件1.txt中
    # file_handle.write(str1)

    pred[pred==1] = 255
    pred[pred==0] = 0
    pred = pred.astype(np.uint8)
    # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    thresh = cv2.Canny(pred, 128, 256)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img = np.copy(image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 绘制矩形
        cv2.rectangle(pred,  (x, y+h), (x+w, y), (255,255,255),-1)
         # 将最小内接矩形填充为白色
    # gray_image = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)  # 如果array已经是灰度数据，则不需要转换
    
    # file_handle=open('2.txt',mode='w')#w 写入模式
    # #将numpy类型转化为list类型
    # x=pred.tolist()
    # #将list转化为string类型
    # strNums=[str(x_i) for x_i in x]
    # str1=",".join(strNums)
    # #将str类型数据存入本地文件1.txt中
    # file_handle.write(str1)
    
    pred[pred==255] = 1
    pred[pred==0] = 0
    
    # file_handle=open('3.txt',mode='w')#w 写入模式
    # #将numpy类型转化为list类型
    # x=pred.tolist()
    # #将list转化为string类型
    # strNums=[str(x_i) for x_i in x]
    # str1=",".join(strNums)
    # #将str类型数据存入本地文件1.txt中
    # file_handle.write(str1)
    
    pred = torch.from_numpy(pred)
    return pred
    
class edge_loss(nn.Module):
  def __init__(self):
    super(edge_loss,self).__init__()

    self.pool=nn.MaxPool2d(3,stride=1,padding=1)
    self.loss=nn.MSELoss().cuda()

  def forward(self,x,target):

    t_smoothed=self.pool(target)
    edge2=torch.abs(target-t_smoothed).cuda()
    edge_loss=self.loss(x,edge2)

    return edge_loss

class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()
    self.sigmoid=nn.Sigmoid()
 
  def forward(self, input, target):
    N = target.size(0)
    smooth = 1e-5
    input=self.sigmoid(input)
 
    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)
 
    intersection = input_flat * target_flat
 
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N
 
    return loss

def seg_loss_calculation(pred_seg, labels):
    bce_logit = nn.BCEWithLogitsLoss().cuda()
    e_loss = edge_loss()
    d_loss = DiceLoss()
    outputs0 = pred_seg[0]
    outputs1 = pred_seg[1]
    outputs2 = pred_seg[2]
    outputs3 = pred_seg[3]
    outputs4 = pred_seg[4]
    outputs5 = pred_seg[5]
    e0 = pred_seg[6]
    e1 = pred_seg[7]
    e2 = pred_seg[8]
    e3 = pred_seg[9]
    e4 = pred_seg[10]

    loss0 = bce_logit(outputs0, labels)
    loss1 = bce_logit(outputs1, labels)
    loss2 = bce_logit(outputs2, labels)
    loss3 = bce_logit(outputs3, labels)
    loss4 = bce_logit(outputs4, labels)

    loss11 = e_loss(e1, labels)
    loss22 = e_loss(e2, labels)
    loss33 = e_loss(e3, labels)
    loss44 = e_loss(e4, labels)
    loss00 = e_loss(e0,labels)

    loss000 = d_loss(outputs0, labels)
    loss111 = d_loss(outputs1, labels)
    loss222 = d_loss(outputs2, labels)
    loss333 = d_loss(outputs3, labels)
    loss444 = d_loss(outputs4, labels)


    loss =  loss0 + loss1 + loss2 + loss3 + loss4 + loss00 + loss11 + loss22 + loss33 + loss44 + loss000 + loss111 + loss222 + loss333 + loss444
    
    return loss, loss000

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def validate_slice(M_enc, A_enc, Seg, Dec, dataloader, args, writer, iter_num):
    training = M_enc.training
    M_enc.eval()
    A_enc.eval()
    Seg.eval()
    Dec.eval()
    # NEW SEG
    bce_logit = nn.BCEWithLogitsLoss()
    #!!!
    # val_dice = AverageMeter()
    seg_val_losses = AverageMeter()
    seg_val_dices = AverageMeter()
    with torch.no_grad():
        for num, sample in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), ncols=80, leave=False)):
            for batch in sample:
                img_val, gt_val = batch['image'].cuda(), batch['label'].cuda()
                
                # file_handle=open('4.txt',mode='w')#w 写入模式
                # #将numpy类型转化为list类型
                # x=gt_val.tolist()
                # #将list转化为string类型
                # strNums=[str(x_i) for x_i in x]
                # str1=",".join(strNums)
                # #将str类型数据存入本地文件1.txt中
                # file_handle.write(str1)
                
                # print(type(gt_val))
                # print(gt_val.shape)
                # index1 = torch.argmax(gt_val)
                # index2 = torch.argmin(gt_val)
                # print(gt_val.view(-1)[index1])
                # print(gt_val.view(-1)[index2])
                # gt_val = gt_val.cpu().numpy()
                # print(type(gt_val))
                # print(gt_val.shape)
                # print(gt_val.max())
                # print(gt_val.min())
                # print(type(img_val))
                # print(type(gt_val))
                # print(img_val.shape)
                # print(img_val.shape)
                with torch.no_grad():
                    a_out = A_enc(img_val)
                    pred_val = Seg(a_out)
                    z_out, mu_out, logvar_out = M_enc(img_val)
                    # for reconstruction
                    reco = Dec(a_out, mu_out)
                pred_val = pred_val[0]
                # pred_val = valid_pred_process(pred_val[0])
                # print(type(pred_val))
                # print(pred_val.shape)
                # pred_val = cpu_tensor.numpy()
                # pred_val = torch.sigmoid(pred_val)
                # seg_loss = seg_loss_calculation(pred_val, gt_val)
                # val_seg.update(seg_loss)
                # val_dice.update(val_dice_class(pred_val.permute(0,2,3,1) > 0.75, gt_val.permute(0,2,3,1), num_class=args.num_classes))
                loss = bce_logit(pred_val, gt_val)
                # pred_val = rectangle_creator(pred_val, args.threshold)
                dice = dice_coef(pred_val, gt_val)
                # seg_val_losses.update(loss.item(), input.size(0))
                # seg_val_dices.update(dice.item(), input.size(0))
                seg_val_losses.update(loss.item(), 1)
                seg_val_dices.update(dice.item(), 1)
                if (num+1) % (args.print_interval+25) == 0:
                    # summarywriter image
                    grid_image = make_grid(img_val.clone().cpu().data, args.batch_size, normalize=True)
                    pred_show = pred_val[0].clone().cpu().data
                    # print(pred_show.shape)
                    # print(pred_val[0].shape)
                    gt_show = gt_val.clone().cpu().data
                    pred_show = np.expand_dims(pred_show,0)
                    # print('-')
                    # print(gt_show.shape)
                    # print(pred_show.shape)
                    writer.add_image('val/images', grid_image, iter_num)
                    writer.add_images('val/ground_truths', gt_show[:, 0:1, :, :], iter_num)
                    writer.add_images('val/prediction', pred_show[:, 0:1, :, :], iter_num)
                    # writer.add_images('val/cup_ground_truths', gt_show[:, 0:1, :, :], iter_num)
                    # writer.add_images('val/cup_preds', pred_show[:, 0:1, :, :], iter_num)
                    # writer.add_images('val/disc_ground_truths', gt_show[:, 1:2, :, :], iter_num)
                    # writer.add_images('val/disc_preds', pred_show[:, 1:2, :, :], iter_num)
    # utils.save_imgs(img.cpu().detach(), gt.cpu().detach(), seg_pred.cpu().detach(), reco.cpu().detach(), img_folder)
    # utils.save_anatomy_factors(a_out[0].cpu().numpy(), anatomy_folder)
    if training:
        M_enc.train()
        A_enc.train()
        Seg.train()
        Dec.train()

    return seg_val_losses.avg, seg_val_dices.avg


def train(model, train_loader, val_loader, writer, args):
    # define the model
    m_encoder = model['M_enc']
    a_encoder = model['A_enc']
    segmentor = model['Seg']
    decoder = model['Dec']
    
    result_dir = os.path.join(args.workspace, 'val_results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_dir = os.path.join(args.workspace, 'models')
    # define the criterion
    # NEW SEG
    bce_logit = nn.BCEWithLogitsLoss()
    d_loss = unary_DiceLoss()

    l1_distance = torch.nn.L1Loss()
    dice_criterion = DiceLoss()
    bce_criterion = torch.nn.BCELoss()
    style_criterion = losses.NTXentLoss(temperature=0.1).cuda()
    
    # # define the optimizer
    # optimizer = optim.Adam([{'params': m_encoder.parameters()}, {'params': a_encoder.parameters()}, 
    #                         {'params': segmentor.parameters()}, {'params': decoder.parameters()}], 
    #                         betas=(0.9, 0.99), lr=args.learning_rate)
    # FOR TWO OPTIMIZER
    optimizer = optim.Adam([{'params': m_encoder.parameters()}, {'params': a_encoder.parameters()}, 
                             {'params': decoder.parameters()}], 
                            betas=(0.9, 0.99), lr=args.learning_rate)
    
    optimizer_seg = optim.SGD([
        {'params': [param for name, param in segmentor.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * 1e-3},
        {'params': [param for name, param in segmentor.named_parameters() if name[-4:] != 'bias'],
         'lr': 1e-3, 'weight_decay': 1e-4}
         ], momentum= 0.9)
    init_state_opt = copy.deepcopy(optimizer_seg.state_dict())
    optimizer_seg.load_state_dict(init_state_opt)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=(16/args.val_interval), verbose=True, min_lr=1e-4)

    # best_val_seg, best_epoch = torch.tensor([0.0, 0.0]), 1
    #  best_val_seg IS LOSS VALUE
    best_seg_dice, best_epoch = 0, 0

    
    
    for epoch in range(args.start_epoch, args.epoches):
        #FOR TWO OPTIMIZER
        if epoch % 40 ==0 &epoch != 0:
            optimizer.param_groups[0]['lr'] = 2 * optimizer.param_groups[0]['lr'] / 10
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / 10
            checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
            for keys, md in models.items():
                pretrained_dict = checkpoint[keys]
                model_dict = md.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                md.load_state_dict(model_dict)
                models[keys] = md
            model.load_state_dict(torch.load(os.path.join(ckpt_path, 'weights','bestmodel.pth')))
        domain_loss = []
        for i in range(len(args.datasetTrain)):
            kl_loss_epoch = AverageMeter()
            seg_loss_epoch = AverageMeter()
            reco_loss_epoch = AverageMeter()
            recoz_loss_epoch = AverageMeter()
            style_loss_epoch = AverageMeter()
            cont_loss_epoch = AverageMeter()
            total_loss_epoch = AverageMeter()
            dice_loss_epoch = AverageMeter()
            domain_loss.append({'kl': kl_loss_epoch, 'seg': seg_loss_epoch, 'reco': reco_loss_epoch, 
                                'recoz': recoz_loss_epoch, 'style': style_loss_epoch, 'cont': cont_loss_epoch, 
                                'total': total_loss_epoch, 'dice': dice_loss_epoch})

        # train in each epoch
        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            iteration = batch_idx + epoch * len(train_loader)
            domain_stylec = [[] for i in range(len(args.datasetTrain))]
            domain_content = [[] for i in range(len(args.datasetTrain))]

            a_encoder.train()
            segmentor.train()
            m_encoder.train()
            decoder.train()
            total_loss = 0
            # total_seg_loss = 0
            for dc, domain in enumerate(sample):
                image = domain['image'].cuda()
                label = domain['label'].cuda()
                # model forward
                a_out = a_encoder(image)

                seg_pred = segmentor(a_out)
              
                z_out, mu_out, logvar_out = m_encoder(image)
                # for reconstruction
                reco = decoder(a_out, z_out)
                z_out_tiled, _, _ = m_encoder(reco)

                # seg_pred = torch.sigmoid(seg_pred)
                # collect style code and anatomy content in each domain
                domain_stylec[dc].append(z_out)
                domain_content[dc].append(a_out)

                # Lank loss for z_out
                reco_loss = l1_distance(reco, image)
                kl_loss = KL_divergence(logvar_out, mu_out)
                # dice_loss = dice_criterion(seg_pred.permute(0,2,3,1), label.permute(0,2,3,1), num_class=args.num_classes)
                # bce_loss = bce_criterion(seg_pred, label)
                # seg_loss = 0.5 * (dice_loss + bce_loss)
                
                
                
                
                seg_loss, dice_loss = seg_loss_calculation(seg_pred, label)
                # bce_loss = bce_logit(seg_pred, label)
                # dice_loss = d_loss(seg_pred, label)
                # seg_loss = 0.5 * (dice_loss + bce_loss)
                
                
                recoz_loss = l1_distance(z_out_tiled, z_out)
                domain_total_loss = args.kl_w * kl_loss + \
                                    args.seg_w * seg_loss + \
                                    args.reco_w * reco_loss + \
                                    args.recoz_w * recoz_loss
                # domain_total_loss = args.kl_w * kl_loss + \
                #                     args.reco_w * reco_loss + \
                #                     args.recoz_w * recoz_loss
                # total_seg_loss += seg_loss
                total_loss += domain_total_loss
                domain_loss[dc]['kl'].update(kl_loss.cpu())
                domain_loss[dc]['seg'].update(seg_loss.cpu())
                domain_loss[dc]['reco'].update(reco_loss.cpu())
                domain_loss[dc]['recoz'].update(recoz_loss.cpu())
                domain_loss[dc]['total'].update(domain_total_loss.cpu())
                domain_loss[dc]['dice'].update(dice_loss.cpu())

                dc_num = len(args.datasetTrain)
                if dc == (dc_num-1):
                    minibatch_domain_stylec = [[] for i in range(dc_num)]
                    domain_label = [[] for i in range(dc_num)]
                    
                    reco_zout = 0
                    for i in range(dc_num):
                        domain_stacked_stylec = torch.cat(domain_stylec[i], dim=0)
                        reco_zout += domain_stacked_stylec * (1-torch.rand(domain_stacked_stylec.size(0),1)*2).cuda()

                        domain_label[i] = torch.tensor([i] * args.n_minibatch).cuda()
                        minibatch_domain_stylec[i] = sample_minibatch_fundus(domain_stacked_stylec, args.n_minibatch, 1)
                        
                    embeddings = torch.cat(minibatch_domain_stylec, dim=0)
                    labels = torch.cat(domain_label, dim=0)
                    style_loss = style_criterion(embeddings, labels)
                    total_loss += args.style_w * style_loss
                    domain_loss[0]['style'].update(style_loss.cpu())

                    domain_content_loss = torch.tensor(0).cuda().float()
                    for i in range(dc_num):
                        domain_stacked_aout = torch.cat(domain_content[i], dim=0)
                        new_reco = decoder(domain_stacked_aout, reco_zout)
                        new_aout = a_encoder(new_reco)
                        domain_content_loss += l1_distance(new_aout, domain_stacked_aout)

                        if (epoch + 1) % (args.print_interval+19) == 0 and (batch_idx % 10) == 0:
                            grid_image = make_grid(new_reco, nrow=args.batch_size, normalize=True)
                            writer.add_image('Train/new_reconstruction', grid_image, epoch)
                            new_aout_shape = new_aout.size()
                            grid_image = make_grid(new_aout.reshape(new_aout_shape[0]*new_aout_shape[1], new_aout_shape[2], new_aout_shape[3]).unsqueeze(dim=1), 
                                                    nrow=args.batch_size, normalize=True)
                            writer.add_image('Train/new_anatomy', grid_image, epoch)

                    total_loss += args.cont_w * (domain_content_loss/dc_num)
                    domain_loss[0]['cont'].update(domain_content_loss.cpu())

                # tensorboard for visualizing train result
                if (epoch + 1) % (args.print_interval+19) ==0 and (batch_idx % 10) == 0:
                    grid_image = make_grid(image, nrow=args.batch_size, normalize=True)
                    writer.add_image('Train/imgs', grid_image, epoch)
                    a_out_shape = a_out.size()
                    grid_image = make_grid(a_out.reshape(a_out_shape[0]*a_out_shape[1], a_out_shape[2], a_out_shape[3]).unsqueeze(dim=1), 
                                            nrow=args.batch_size, normalize=True)
                    writer.add_image('Train/anatomy', grid_image, epoch)
                    grid_image = make_grid(reco, nrow=args.batch_size, normalize=True)
                    writer.add_image('Train/reconstruction', grid_image, epoch)
                    label_shape = label.size()
                    grid_image = make_grid(label.reshape(label_shape[0]*label_shape[1], label_shape[2], label_shape[3]).unsqueeze(dim=1), 
                                            nrow=args.batch_size, normalize=True)
                    writer.add_image('Train/mask', grid_image, epoch)
                    pred_shape = seg_pred[0].size()
                    grid_image = make_grid(seg_pred[0].reshape(pred_shape[0]*pred_shape[1], pred_shape[2], pred_shape[3]).unsqueeze(dim=1), 
                                            nrow=args.batch_size, normalize=True)
                    writer.add_image('Train/prediction', grid_image, epoch)
            
            # backward the gradient
            total_loss = total_loss / len(args.datasetTrain)
            optimizer.zero_grad()
            optimizer_seg.zero_grad()
            total_loss.backward()
            optimizer.step()
            # total_seg_loss.backward()
            optimizer_seg.step()
            
        # print training result
        logging.info('\n Epoch[%4d/%4d]-Lr: %.6f --> Train...' % (epoch+1, args.epoches, optimizer.param_groups[0]['lr']))
        for i in range(len(args.datasetTrain)):
            logging.info('\t Domain-%d: [Total Loss: %.4f]: KL Loss = %.4f, Seg Loss = %.4f, Reco Loss = %.4f, RecoZ Loss = %.4f, Dice Loss = %.4f' %
                        (args.datasetTrain[i], domain_loss[i]['total'].avg, domain_loss[i]['kl'].avg, domain_loss[i]['seg'].avg, 
                        domain_loss[i]['reco'].avg, domain_loss[i]['recoz'].avg, domain_loss[i]['dice'].avg))
        logging.info('\t Domain-all: Domain style contrast Loss = %.4f, Domain content Loss = %.4f' % 
                    (domain_loss[0]['style'].avg, domain_loss[0]['cont'].avg))

        # tensorboard
        if (epoch+1) % args.print_interval == 0:
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('Train/Domain_all/Losses', 
                        {'style': domain_loss[0]['style'].avg, 'cont': domain_loss[0]['cont'].avg}, epoch)
            for i, train_dc in enumerate(args.datasetTrain):
                writer.add_scalars('Train/Domain{}/Losses'.format(train_dc), 
                        {'kl': domain_loss[i]['kl'].avg, 'seg': domain_loss[i]['seg'].avg, 
                        'reco': domain_loss[i]['reco'].avg, 'recoz': domain_loss[i]['recoz'].avg,
                        'total': domain_loss[i]['total'].avg, 'dice':domain_loss[i]['dice'].avg}, epoch)

        # validate and visualization
        
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if (epoch + 1) % args.val_interval == 0:
            val_img_path = os.path.join(result_dir, 'Ep_%04d_imgs.png' % (epoch + 1))
            val_anatomy_path = os.path.join(result_dir, 'Ep_%04d_anatomys.png' % (epoch + 1))
            val_seg_loss, val_seg_dice = validate_slice(m_encoder, a_encoder, segmentor, decoder, val_loader, args, writer, epoch)
            val_seg = (val_seg_loss+val_seg_dice)/2
            # val_dice = validate_slice(m_encoder, a_encoder, segmentor, decoder, val_loader, val_img_path,
            #                             val_anatomy_path, args, writer, epoch)
            logging.info('\n Epoch[%4d/%4d] --> Valid...' % (epoch + 1, args.epoches))
            logging.info(
                '\t [Seg Coef:mean=%.4f|loss:mean=%.4f|dice:mean=%.4f]' % (val_seg,val_seg_loss,val_seg_dice))
            writer.add_scalars('Val/Seg', {'seg': val_seg}, epoch)
            # save best model
            if best_seg_dice <= val_seg_dice:
                best_model_path = os.path.join(model_dir, 'best_model.pth')
                torch.save(
                    {'M_enc': m_encoder.state_dict(), 'A_enc': a_encoder.state_dict(), 'Seg': segmentor.state_dict(),
                     'Dec': decoder.state_dict()}, best_model_path)
                logging.info(
                    '\n Epoch[%4d/%4d] --> Seg dice improved from %.4f (seg=%.4f in epoch %4d) to %.4f (seg=%.4f)' %
                    (epoch + 1, args.epoches, best_seg_dice, best_seg_dice, best_epoch, val_seg_dice, val_seg_dice))
                best_seg_dice, best_epoch = val_seg_dice, epoch + 1
            else:
                logging.info('\n Epoch[%4d/%4d] -->Seg dice did not improved with %.4f (seg=%.4f in epoch %d)' %
                             (epoch + 1, args.epoches, best_seg_dice, best_seg_dice, best_epoch))
            # check for plateau
            scheduler.step(val_seg)

        # if (epoch+1) % args.val_interval == 0:
        #     val_img_path = os.path.join(result_dir, 'Ep_%04d_imgs.png' % (epoch+1))
        #     val_anatomy_path = os.path.join(result_dir, 'Ep_%04d_anatomys.png' % (epoch+1))
        #     val_dice = validate_slice(m_encoder, a_encoder, segmentor, decoder, val_loader, args, writer, epoch)
        #     # val_dice = validate_slice(m_encoder, a_encoder, segmentor, decoder, val_loader, val_img_path,
        #     #                             val_anatomy_path, args, writer, epoch)
        #     logging.info('\n Epoch[%4d/%4d] --> Valid...' % (epoch+1, args.epoches))
        #     logging.info('\t [Dice Coef: mean=%.4f, cup=%.4f, disc=%.4f]' % (torch.mean(val_dice), val_dice[0], val_dice[1]))
        #     writer.add_scalars('Val/Dice', {'cup': val_dice[0], 'disc': val_dice[1],
        #                         'mean': torch.mean(val_dice)}, epoch)
        #     # save best model
        #     if torch.mean(val_dice) >= torch.mean(best_val_dice):
        #         best_model_path = os.path.join(model_dir, 'best_model.pth')
        #         torch.save({'M_enc': m_encoder.state_dict(), 'A_enc': a_encoder.state_dict(), 'Seg': segmentor.state_dict(),
        #                     'Dec': decoder.state_dict()}, best_model_path)
        #         logging.info('\n Epoch[%4d/%4d] --> Dice improved from %.4f (cup=%.4f, disc=%.4f in epoch %4d) to %.4f (cup=%.4f, disc=%.4f)' %
        #             (epoch+1, args.epoches, torch.mean(best_val_dice), best_val_dice[0], best_val_dice[1], best_epoch, torch.mean(val_dice),
        #             val_dice[0], val_dice[1]))
        #         best_val_dice, best_epoch = val_dice, epoch+1
        #     else:
        #         logging.info('\n Epoch[%4d/%4d] --> Dice did not improved with %.4f (cup=%.4f, disc=%.4f in epoch %d)' %
        #                     (epoch+1, args.epoches, torch.mean(best_val_dice), best_val_dice[0], best_val_dice[1], best_epoch))
        #     # check for plateau
        #     scheduler.step(torch.mean(val_dice))

        # # save images
        # train_img_dir = utils.check_folder(os.path.join(args.exp_dir, 'train_img'))
        # train_img_path = os.path.join(train_img_dir, 'Ep_%04d_imgs_dice_%.4f.png' % (epoch, val_dice))
        # utils.save_imgs(img.cpu().detach(), gt.cpu().detach(), seg_pred.cpu().detach(), reco.cpu().detach(),
        #                 train_img_path)
        # train_anatomy_path = os.path.join(train_img_dir, 'Ep_%04d_anatomys_dice_%.4f.png' % (epoch, val_dice))
        # utils.save_anatomy_factors(a_out[0].cpu().detach(), train_anatomy_path)

        # save model
        if (epoch+1) >= (args.epoches-100) and (epoch+1) % args.save_interval == 0:
            model_path = os.path.join(model_dir, 'Ep_%04d_dice_%.4f.pth' % ((epoch+1), val_seg))
            torch.save({'M_enc': m_encoder.state_dict(), 'A_enc': a_encoder.state_dict(), 'Seg': segmentor.state_dict(), 
                        'Dec': decoder.state_dict(), 'optim': optimizer.state_dict()}, model_path)
            logging.info('\t [Save Model] to %s' % model_path)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    # define logger
    now = datetime.now()
    args.workspace = os.path.join(args.exp_dir, 'train-domain'+str(args.datasetTest), now.strftime('%Y%m%d_%H%M%S.%f'))
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)
    logging.basicConfig(filename=os.path.join(args.workspace, 'train.log'), level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # print all parameters
    for name, v in vars(args).items():
        logging.info(name + ': ' + str(v))
    with open(os.path.join(args.workspace, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        # tr.RandomCrop(512),
        # tr.RandomRotate(),
        # tr.RandomFlip(),
        # tr.elastic_transform(),
        # tr.add_salt_pepper_noise(),
        # tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        # tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    # dataloader config
    # print(args.datasetTrain,'!!!')
    # print(args.data_dir, '!!!')
    train_set = BreastSegmentation(base_dir=args.data_dir, phase='train', splitid=args.datasetTrain,
                                        transform=composed_transforms_tr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_set = BreastSegmentation(base_dir=args.data_dir, phase='valid', splitid=args.datasetTest,
                                        transform=composed_transforms_ts)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=6)
    
    # test_set = BreastSegmentation(base_dir=args.data_dir, phase='finaltest', splitid=args.datasetTest,
    #                                     transform=composed_transforms_ts)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6)

    # model configuration
    m_encoder = MEncoder(z_length=args.z_length, in_channel=args.in_channel, img_size=args.data_size)
    a_encoder = AEncoder(in_channel=args.in_channel, width=256, height=256, ndf=16, num_output_channels=args.anatomy_channel, norm='batchnorm', upsample='bilinear')
    segmentor = DeepLab(num_classes=args.num_classes,
                  backbone='daf_ds',
                  output_stride=16,
                  mm='dgnlb',
                  sync_bn='auto',
                  freeze_bn=False).cuda()
    # segmentor.load_state_dict(torch.load('./GlobalGuidanceNet/Results/assemble-USG/weights/bestmodel.pth'))
    decoder = Ada_Decoder(anatomy_out_channel=args.anatomy_channel, z_length=args.z_length, out_channel=args.in_channel) 
    print('parameter numer:', sum([p.numel() for p in m_encoder.parameters()]+
                                    [p.numel() for p in a_encoder.parameters()]+
                                    [p.numel() for p in segmentor.parameters()]+
                                    [p.numel() for p in decoder.parameters()]))
    models = {'M_enc': m_encoder.cuda(), 'A_enc': a_encoder.cuda(), 'Seg': segmentor.cuda(), 'Dec': decoder.cuda()}
#     # IF NECESSARY
#     checkpoint = torch.load('output/cddsa/breastultrasound/train-domain[5]/20240420_192923.987180/models/best_model.pth')
#     for keys, md in models.items():
#         pretrained_dict = checkpoint[keys]
#         model_dict = md.state_dict()
#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 3. load the new state dict
#         md.load_state_dict(model_dict)
#         models[keys] = md
        
    if args.resume:
        checkpoint = torch.load(args.resume)
        for keys, md in models.items():
            pretrained_dict = checkpoint[keys]
            model_dict = md.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            md.load_state_dict(model_dict)
            models[keys] = md
        print('Resume models finished!')
    else:
        for md in models.values():
            initialize_weights(md, args.weight_init)

    # summary writer config
    run_dir = os.path.join(args.workspace, 'run')
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    writer = SummaryWriter(log_dir=run_dir, comment=args.exp_dir.split('/')[-1])

    # train
    train(models, train_loader, valid_loader, writer, args)

if __name__ == '__main__':
    main()
