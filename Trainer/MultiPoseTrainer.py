from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from torch.nn import utils

import numpy as np
from Net.Losses import FocalLoss, RegL1Loss,RegWeightedL1Loss
from utils.utils_Decode import _sigmoid
from utils.utils import gen_oracle_map,AverageMeter
import time
import torch
from progress.bar import Bar  # 显示进度条
from Detector.Decoder import multi_pose_decode,multi_pose_post_process
from Trainer.Drawer import COCO_Drawer

class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats



class MultiPoseLoss(torch.nn.Module):
  def __init__(self,opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss
    self.crit_kp = RegWeightedL1Loss()
    self.crit_reg = RegL1Loss
    self.opt = opt

  def forward(self, output, batch):
    """
    :param output:   output 为网络输出的六个头的列表
    :param batch:    为Ground_Truth
    :return:
    """
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
    
    if isinstance(output,list):
      output = output[-1]

    output['hm'] = _sigmoid(output['hm'])
    output['hm_hp'] = _sigmoid(output['hm_hp'])

    # if opt.eval_oracle_hmhp:
    #   output['hm_hp'] = batch['hm_hp']
    # if opt.eval_oracle_hm:
    #   output['hm'] = batch['hm']
    # if opt.eval_oracle_kps:
    #   output['hps'] = torch.from_numpy(gen_oracle_map(
    #     batch['hps'].detach().cpu().numpy(),
    #     batch['ind'].detach().cpu().numpy(),
    #     opt.output_res, opt.output_res)).to(opt.device)
    # if opt.eval_oracle_hp_offset:
    #   output['hp_offset'] = torch.from_numpy(gen_oracle_map(
    #     batch['hp_offset'].detach().cpu().numpy(),
    #     batch['hp_ind'].detach().cpu().numpy(),
    #     opt.output_res, opt.output_res)).to(opt.device)


    hm_loss += self.crit(output['hm'], batch['hm'])

    hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                              batch['ind'], batch['hps'])

    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                               batch['ind'], batch['wh'])

    off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                              batch['ind'], batch['reg'])

    hp_offset_loss += self.crit_reg(output['hp_offset'], batch['hp_mask'],
                                    batch['hp_ind'], batch['hp_offset'])

    hm_hp_loss += self.crit(output['hm_hp'], batch['hm_hp'])

    loss = hm_loss + 0.1*wh_loss + off_loss + hp_loss + \
           hm_hp_loss + hp_offset_loss
    
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss, 
                  'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class MultiPoseTrainer(object):
  def __init__( self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)
    # outputs = self.model(batch['input'])
    # loss, loss_stats = self.loss(outputs, batch)

  def set_device(self, device):  # 设置训练设备
    self.model_with_loss = self.model_with_loss.to(device)
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss', 
                   'hp_offset_loss', 'wh_loss', 'off_loss']
    loss = MultiPoseLoss(opt)
    return loss_states, loss

  def run_epoch(self, train_or_val, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if train_or_val == 'train':
      model_with_loss.train()
    else:     # 如果是验证集
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}

    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}

    num_iters = len(data_loader)
    bar = Bar('{}/{}'.format('multi_pose', 'train'), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:  # batch为ground truth字典
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)  # 转为gpu上

      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if train_or_val == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=train_or_val,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter >0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format('multi_pose', 'default', Bar.suffix))
      else:
        bar.next()

      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results

  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] 
    hm_hp = output['hm_hp']
    hp_offset = output['hp_offset']
    # 解码器
    dets = multi_pose_decode(
      output['hm'], output['wh'], output['hps'], 
      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    dets[:, :, :4] *= opt.down_ratio
    dets[:, :, 5:39] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    dets_gt[:, :, 5:39] *= opt.down_ratio
    for i in range(1):
      debugger = COCO_Drawer()
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')
          debugger.add_coco_hp(dets[i, k, 5:39], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')
          debugger.add_coco_hp(dets_gt[i, k, 5:39], img_id='out_gt')


      pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
      debugger.add_blend_img(img, gt, 'gt_hmhp')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        debugger.show_all_imgs(pause=False)
  def save_result(self, output, batch, results):
    # down_ratio = opt.input_res / opt.output_res
    hm,wh,hps = output['hm'], output['wh'], output['hps']
    reg = output['reg']
    hm_hp = output['hm_hp']
    hp_offset = output['hp_offset']
    dets = multi_pose_decode(
      hm,wh,hps, reg, hm_hp, hp_offset, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
  
    dets_out = multi_pose_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
