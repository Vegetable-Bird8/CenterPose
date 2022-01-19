from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

# from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from DataLoader.multi_pose import COCOHP
from Detector.Multi_Pose_Detecter import MultiPoseDetector
# opt = opts().parse()
# save
def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  Dataset = COCOHP
  # print(opt)
  Logger(opt)
  Detector = MultiPoseDetector
  
  # split = 'val' if not opt.trainval else 'test'
  split = 'val'
  dataset = Dataset(opt, split)
  detector = Detector(opt,train_model='/home/hsn/CenterNet/hsn/multi_pose/save_model/resnet_18_6.pth')  # initialition

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  coco_eval_result = dataset.run_eval(results, opt.save_dir)
  print(f"read from coco eval:{len(coco_eval_result)},type is {type(coco_eval_result)},'\n====================>'{coco_eval_result}")
  return coco_eval_result
if __name__ == '__main__':
  opt = opts().parse()
  # if opt.not_prefetch_test:
  test(opt)
  # else:
  #   prefetch_test(opt)