# under windows
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CenterPose -> post_process
@IDE    ：PyCharm
@Author ：SN.Han
@Date   ：2021/12/21 17:12
@Desc   ：
=================================================='''
from utils.image import transform_preds
import numpy as np

def multi_pose_post_process(dets, c, s, h, w):
  """
  :param dets:    检测框  shape[1,batch*K,40]
  :param c:       中心点[x,y]
  :param s:       max[h,w]
  :param h:       输出图像的h
  :param w:       输出图像的w
  :return:        对所有坐标进行仿射变换
  """
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  # ret = []
  for i in range(dets.shape[0]):  # 对batch进行循环
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    # ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})  # {1:top_preds]
  return top_preds