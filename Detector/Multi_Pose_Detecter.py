from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
from Detector.Decoder import multi_pose_decode,multi_pose_post_process
from Net.MobileNet_v2 import get_MobileNet
from Net.resnet_dcn import get_pose_net
from model import load_model
from utils.image import get_affine_transform
from Trainer.Drawer import COCO_Drawer
from DataLoader.multi_pose import COCO_MEAN,COCO_STD,FLIP_IDX

class MultiPoseDetector(object):
  def __init__(self, opt,train_model = ''):
    self.flip_idx = FLIP_IDX
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    # 创建模型
    print('Creating model...')
    # self.model = get_MobileNet()
    self.model = get_pose_net(18)
    if train_model =='':
      self.model = load_model(self.model, opt.load_model)
    else:
      self.model = load_model(self.model,train_model)    
    self.model = self.model.to(opt.device)
    self.model.eval()  #eval模式


    self.mean = np.array(COCO_MEAN, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(COCO_STD, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = 1
    self.scales = 1
    self.opt = opt
    self.pause = True

  """
  图像预处理

  """
  def pre_process(self, image, scale, meta=None):   # 图像预处理
    height, width = image.shape[0:2]     # 读取高和宽
    new_height = int(height * scale)      # scale 为缩放系数
    new_width  = int(width * scale)
    # # fix_res
    # inp_height, inp_width = self.opt.input_h, self.opt.input_w
    # c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    # s = max(height, width) * 1.0
    # keep res
    
    # 这部分代码作用就是通过按位或运算，找到最接近的2的倍数-1作为最终的尺度。
    inp_height = (new_height | self.opt.pad) + 1  # 等同于 inp_height = new_height+pad +1 if new_height>pad else pad+1
    inp_width = (new_width | self.opt.pad) + 1
    c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
    s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])   # 根据输入图片获得仿射矩阵
    resized_image = cv2.resize(image, (new_width, new_height))     # 改变图像大小
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),    # 仿射变换
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)  # 归一化
    # cv 读取的图像 通道排列为 （h,w,c)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 通道数变在前 并变为[1,3,h,w]

    images = torch.from_numpy(images)  # 转为tensor
    meta = {'c': c, 's': s,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta


  def process(self, images, return_time=False):  # 推理+解码
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]  #取出字典类型的结果
      output['hm'] = output['hm'].sigmoid_()
      output['hm_hp'] = output['hm_hp'].sigmoid_()
      # 加载完数据，以下为推理
      torch.cuda.synchronize()
      forward_time = time.time()    # 推理时间

      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg = output['reg'], hm_hp = output['hm_hp'], hp_offset = output['hp_offset'],
        K=self.opt.K)
      # print
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  # def post_process(self, dets, meta, scale=1):  #后处理 将推理出来的高分图像进行解码显示
  #   dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])  # shape [1,batch*k,40]
  #   top_k_pred = multi_pose_post_process(
  #     dets.copy(), [meta['c']], [meta['s']],
  #     meta['out_height'], meta['out_width'])

  #   top_k_pred = np.array(top_k_pred, dtype=np.float32).reshape(-1, 39)
  #   top_k_pred[:, :4] /= scale
  #   top_k_pred[:, 5:] /= scale
  #   return top_k_pred # 字典
  
      """    
    detections shape [batch,K,40]
    detections[0:4]  为 bboxes
    detections[4]    为 scores
    detections[5:39] 为 hps 单数为x 双数为y
    detections[39]   为 clses 类别
    """
  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  # def merge_outputs(self, detections):  # detections[1] =top_k_pred
  #   results = {}
  #   results[1] = np.concatenate(
  #       [detection for detection in detections], axis=0).astype(np.float32)
  #   # if self.opt.nms or len(self.opt.test_scales) > 1:
  #   #   soft_nms_39(results[1], Nt=0.5, method=2)
  #   results[1] = results[1].tolist()
  #   return results
  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    # if self.opt.nms or len(self.opt.test_scales) > 1:
    #   soft_nms_39(results[1], Nt=0.5, method=2)
    results[1] = results[1].tolist()
    return results

  def debug(self, drawer, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)  #恢复opencv的通道排列
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)   # 通过均值和方差恢复图像色彩
      
    pred = drawer.gen_colormap(output['hm'][0].detach().cpu().numpy())
    drawer.add_blend_img(img, pred, 'pred_hm')
    
    pred = drawer.gen_colormap_hp(
      output['hm_hp'][0].detach().cpu().numpy())
    drawer.add_blend_img(img, pred, 'pred_hmhp')

  def show_results(self, drawer, image, results):
    drawer.add_img(image, img_id='multi_pose')
    for bbox in results[1]:
      if bbox[4] > self.opt.vis_thresh:   # self.opt.vis_thresh == 0.3 
        drawer.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        drawer.add_coco_hp(bbox[5:39], img_id='multi_pose')
    drawer.show_all_imgs(pause=self.pause)

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0

    drawer = COCO_Drawer()
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):  # 如果输入的是图片
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type(''):  # 如果输入的是路径
      image = cv2.imread(image_or_path_or_tensor)
    else:  # 如果是Tensor
      image = image_or_path_or_tensor['image'][0].numpy()  # 转为numpy() 此时不需要预处理
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True

    loaded_time = time.time()
    load_time += (loaded_time - start_time)

    detections = []
    # for scale in self.scal:
    scale = self.scales
    scale_start_time = time.time()
    if not pre_processed:
      images, meta = self.pre_process(image, scale, meta)
    else:
      # import pdb; pdb.set_trace()
      images = pre_processed_images['images'][scale][0]
      meta = pre_processed_images['meta'][scale]
      meta = {k: v.numpy()[0] for k, v in meta.items()}
    images = images.to(self.opt.device)
    torch.cuda.synchronize()
    pre_process_time = time.time()
    pre_time += pre_process_time - scale_start_time
    
    output, dets, forward_time = self.process(images, return_time=True)

    torch.cuda.synchronize()
    net_time += forward_time - pre_process_time
    decode_time = time.time()
    dec_time += decode_time - forward_time
    
    if self.opt.debug >= 2:
      self.debug(drawer, images, dets, output, scale)
    
    dets = self.post_process(dets, meta, scale)
    torch.cuda.synchronize()
    post_process_time = time.time()
    post_time += post_process_time - decode_time

    detections.append(dets)

    results = self.merge_outputs(detections)  #返回字典
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(drawer, image, results)

    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}
    # tot 为初始时间
    # load 为加载图片时间
    # pre 为预处理图片时间
    # net 为模型推理时间
    # dec 为解码时间
    # post 为后处理时间
    # merge 为整理结果时间