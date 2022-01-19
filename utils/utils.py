# under windows
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CenterPose -> utils
@IDE    ：PyCharm
@Author ：SN.Han
@Date   ：2021/12/21 8:30
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
from collections import OrderedDict
import numba
import numpy as np

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    if self.count > 0:
      self.avg = self.sum / self.count


def flip_tensor(x):
  return torch.flip(x, [3])   # 镜像翻转
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()  #该方法主要用于将cpu上的tensor转为numpy数据，并镜像翻转
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def load_model(model, pretrain_dir):
  state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
  print('loaded pretrained weights form %s !' % pretrain_dir)
  state_dict = OrderedDict()

  # convert data_parallal to model
  for key in state_dict_:
    if key.startswith('module') and not key.startswith('module_list'):
      state_dict[key[7:]] = state_dict_[key]
    else:
      state_dict[key] = state_dict_[key]

  # check loaded parameters and created model parameters
  model_state_dict = model.state_dict()
  for key in state_dict:
    if key in model_state_dict:
      if state_dict[key].shape != model_state_dict[key].shape:
        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
          key, model_state_dict[key].shape, state_dict[key].shape))
        state_dict[key] = model_state_dict[key]
    else:
      print('Drop parameter {}.'.format(key))
  for key in model_state_dict:
    if key not in state_dict:
      print('No param {}.'.format(key))
      state_dict[key] = model_state_dict[key]
  model.load_state_dict(state_dict, strict=False)

  return model


def count_parameters(model):
  num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
  print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=512):
  flops = []
  handles = []

  def conv_hook(self, input, output):
    flops.append(output.shape[2] ** 2 *
                 self.kernel_size[0] ** 2 *
                 self.in_channels *
                 self.out_channels /
                 self.groups / 1e6)

  def fc_hook(self, input, output):
    flops.append(self.in_features * self.out_features / 1e6)

  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      handles.append(m.register_forward_hook(conv_hook))
    if isinstance(m, nn.Linear):
      handles.append(m.register_forward_hook(fc_hook))

  with torch.no_grad():
    _ = model(torch.randn(1, 3, input_size, input_size))
  print("Total FLOPs = %f M" % sum(flops))

  for h in handles:
    h.remove()



@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):
  # feat: B x maxN x featDim
  # ind: B x maxN
  batch_size = feat.shape[0]
  max_objs = feat.shape[1]
  feat_dim = feat.shape[2]
  out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
  vis = np.zeros((batch_size, h, w), dtype=np.uint8)
  ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for i in range(batch_size):
    queue_ind = np.zeros((h*w*2, 2), dtype=np.int32)
    queue_feat = np.zeros((h*w*2, feat_dim), dtype=np.float32)
    head, tail = 0, 0
    for j in range(max_objs):
      if ind[i][j] > 0:
        x, y = ind[i][j] % w, ind[i][j] // w
        out[i, :, y, x] = feat[i][j]
        vis[i, y, x] = 1
        queue_ind[tail] = x, y
        queue_feat[tail] = feat[i][j]
        tail += 1
    while tail - head > 0:
      x, y = queue_ind[head]
      f = queue_feat[head]
      head += 1
      for (dx, dy) in ds:
        xx, yy = x + dx, y + dy
        if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
          out[i, :, yy, xx] = f
          vis[i, yy, xx] = 1
          queue_ind[tail] = xx, yy
          queue_feat[tail] = f
          tail += 1
  return out