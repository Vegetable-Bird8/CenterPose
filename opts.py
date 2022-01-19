from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
# flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # Training settings
    self.parser.add_argument('--data_dir', type=str, default='/home/hsn/CenterNet/hsn/CenterPose/Dataset/COCO Dataset2017',help="Data directory")


    self.parser.add_argument('--exp_id', default='multi_pose')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')

    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=8, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=512, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=512, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=512, 
                             help='input width. -1 for default from dataset.')
    self.parser.add_argument('--output_res', type=int, default=128, 
                             help='input width. -1 for default from dataset.')
    # Trainer
    self.parser.add_argument('--lr', type=float, default=5e-4, 
                             help='learning rate for batch size 32.')

    self.parser.add_argument('--lr_step', type=str, default='70,90',
                             help='drop learning rate by 10.')

    self.parser.add_argument('--num_epochs', type=int, default=100,
                             help='total training epochs.')

    self.parser.add_argument('--batch_size', type=int, default=1,
                             help='batch size')
    self.parser.add_argument('--val_intervals', type=int, default=1,
                            help='number of epochs to run validation.')

    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 

    # dataset
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    # multi_pose
    self.parser.add_argument('--center_thresh', type=float, default=0.1,
                          help='threshold for centermap.')
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    
    # ground truth validation
    self.parser.add_argument('--eval_oracle_hm', action='store_true', 
                             help='use ground center heatmap.')
    self.parser.add_argument('--eval_oracle_wh', action='store_true', 
                             help='use ground truth bounding box size.')
    self.parser.add_argument('--eval_oracle_offset', action='store_true', 
                             help='use ground truth local heatmap offset.')
    self.parser.add_argument('--eval_oracle_kps', action='store_true', 
                             help='use ground truth human pose offset.')
    self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    self.parser.add_argument('--eval_oracle_dep', action='store_true', 
                             help='use ground truth depth.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

    # if opt.head_conv == -1: # init default head_conv
    #   opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 31
    # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    # if opt.trainval:
    #   opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..',)
    opt.exp_dir = os.path.join(opt.root_dir, 'multi_pose')
    opt.save_dir = os.path.join(opt.exp_dir, 'save_model')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    opt.pad = 31
    return opt

