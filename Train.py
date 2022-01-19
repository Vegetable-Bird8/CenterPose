# under windows
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CenterPose -> Train
@IDE    ：PyCharm
@Author ：SN.Han
@Date   ：2021/12/23 14:04
@Desc   ：
=================================================='''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from json import decoder

import torch
import os
from model import save_model
from logger import Logger
from Net.MobileNet_v2 import get_MobileNet
from Net.resnet_dcn import get_pose_net
from DataLoader.multi_pose import COCOHP
from Trainer.MultiPoseTrainer import MultiPoseTrainer
from opts import opts
from model import load_model
from Detector.Multi_Pose_Detecter import MultiPoseDetector
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')



def main(opt):
  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = True    # disable this if OOM at beginning of training
  Dataset = COCOHP   # 初始化类
  # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print('Setting up data...')
  train_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'train'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
    drop_last=True
  )
  val_dataset = Dataset(opt, 'val')
  val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
  )


  print('Creating model...')
  model = get_MobileNet(train_or_test='train')
  # model = get_pose_net(50)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  # 用来继续训练
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
# 训练器
  Trainer = MultiPoseTrainer
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(device)
 # 初始化验证器
  Detector = MultiPoseDetector





  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))

    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      logger.write('\nvalidation performence:==========>\n')
      save_path = os.path.join(opt.save_dir, 'model_{}.pth'.format(mark))
      save_model(save_path,epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        val_num_iters = len(val_dataset)
        results = {}
        detector = Detector(opt,train_model=save_path)
        # bar = Bar('{}'.format(opt.exp_id), max=num_iters)
        # time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
        # avg_time_stats = {t: AverageMeter() for t in time_stats}
        for ind in range(val_num_iters):
          img_id = val_dataset.images[ind]
          img_info = val_dataset.coco.loadImgs(ids=[img_id])[0]
          img_path = os.path.join(val_dataset.img_dir, img_info['file_name'])
          ret = detector.run(img_path)
          
          results[img_id] = ret['results']

          # Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
          #                ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        #   for t in avg_time_stats:
        #     avg_time_stats[t].update(ret[t])
        #     Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        #   bar.next()
        # bar.finish()
        coco_eval_results = val_dataset.run_eval(results, opt.save_dir)
        print(coco_eval_results)
        logger.scalar_summary('keypoints val_mAP/mAP', coco_eval_results[0], epoch)
        logger.scalar_summary('bbox val_mAP/mAP', coco_eval_results[1], epoch)
        logger.write('keypoints val_mAP/mAP :{}'.format(coco_eval_results[0]))
        logger.write('bbox val_mAP/mAP :{}'.format(coco_eval_results[1]))
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))    # 不同的损失函数之间使用 | 隔开


      if log_dict_val['loss'] < best:   # 如果损失更小，则存储损失更小的模型 
        best = log_dict_val['loss']
        save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                   epoch, model)
      
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    logger.write('\n')
    
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
  logger.close()


if __name__ == '__main__':
  opt = opts().parse()
  os.chdir(opt.root_dir)

  # opt.log_dir = os.path.join(opt.root_dir, 'logs', opt.log_name)
  # opt.ckpt_dir = os.path.join(opt.root_dir, 'ckpt', opt.log_name)
  # # opt.pretrain_dir = os.path.join(opt.root_dir, 'ckpt', opt.pretrain_name, 'checkpoint.t7')
  #
  # os.makedirs(opt.log_dir, exist_ok=True)
  # os.makedirs(opt.ckpt_dir, exist_ok=True)
  # # 生成权重缩减步数列表
  # opt.lr_step = [int(s) for s in opt.lr_step.split(',')]
  main(opt)