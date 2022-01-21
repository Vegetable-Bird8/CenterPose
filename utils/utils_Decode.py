# under windows
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CenterPose -> utils
@IDE    ：PyCharm
@Author ：SN.Han
@Date   ：2021/12/18 10:20
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from PIL import Image

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    # hmax = nn.functional.max_pool2d(
    # heat, (kernel, kernel), stride=1, padding=pad)
    Max_pool = nn.MaxPool2d(kernel,stride=1, padding=pad)
    hmax = Max_pool(heat)
    keep = (hmax == heat).float()
    return heat * keep

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # ind 输入的时候比feat少一个维度，此处增加一个维度和feat相等
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
# 这里没有断开contiguous
def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3)) 
    feat = _gather_feat(feat, ind)
    return feat
# 此函数是 _topk函数的简化版本
# 不再将所有通道放在一起比较
def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)  
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

# 本质上就是获得前K个索引值以及对应的scores
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    # topk 函数输出(batch,cat,k)的scores和indx
    # indx的每个值 = y*width +x 从而可以解析出scores对应点的 x y
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)  # 这步有点多余
    topk_ys = (topk_inds / width).int().float()  # 解析出对应的 x y 坐标
    topk_xs = (topk_inds % width).int().float()
    # 下面再找所有类别中的前K个值 
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # topl_score,topk_ind shape = [b,k]
    topk_clses = (topk_ind / K).int()   # 最大值对应的类别
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        # 通过这一步操作 ， 返回的topk_inds为多个通道之间的索引值，搭配topk_clses
        # shape = [b,k]
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


