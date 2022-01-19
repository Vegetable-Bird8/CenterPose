from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torch
import cv2
import math
import numpy as np
import json
import os
from torch.utils import data

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.image import color_aug, get_border,\
                        get_affine_transform, affine_transform,\
                        gaussian_radius, draw_gaussian


COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]

EDGES = [[0, 1], [0, 2], [1, 3], [2, 4],
         [4, 6], [3, 5], [5, 6],
         [5, 7], [7, 9], [6, 8], [8, 10],
         [6, 12], [5, 11], [11, 12],
         [12, 14], [14, 16], [11, 13], [13, 15]]  # 可以连接的点

ACC_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 除了鼻子之外的点

FLIP_IDX = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
            [11, 12], [13, 14], [15, 16]]  # 对称的点


class COCOHP(data.Dataset):
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt,split):  # opt = parser.parse_args()包括 data_dir,split,rand_crop
        super(COCOHP, self).__init__()
        self.num_classes = 1
        self.num_joints = 17
        self._data_rng = np.random.RandomState(123)  # 随机种子
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)  #特征值
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)# 特征向量
        self.mean = np.array(COCO_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(COCO_STD, dtype=np.float32).reshape(1, 1, 3)
        self.edges = EDGES  # 可以连接的点
        self.acc_idxs = ACC_IDXS  # 除了鼻子之外的点
        self.opt = opt
        self.split = split  #train or test
        self.data_dir = opt.data_dir  # 文件的根目录 包含annotations和img
        self.flip_idx = FLIP_IDX  # 镜像对称点

        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(self.split))
        if self.split == 'test':  # 待下载
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(self.split)
        else:   # 验证集
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'person_keypoints_{}2017.json').format(self.split)
        self.max_objs = 32

        print('==> initializing coco 2017 {} data.'.format(split))
        # 读取 coco标签数据
        self.coco = coco.COCO(self.annot_path)
        # 返回所有图片的id
        image_ids = self.coco.getImgIds()

        if self.split == 'train':
            self.images = []
            for img_id in image_ids:
                # 读取标签id，如果对应有标签，就将他放入图片列表中
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids  # 此时images为一片序号
        self.num_samples = len(self.images)  # 图片的数量
        print('Loaded {} {} samples'.format(self.split, self.num_samples))

    """    
    # class MultiPoseDataset(data.Dataset):
    # coco box 默认内容为 [x1,y1,w,h] ,需要转变为 [x1,y1,x2,y2]
    """
    def _coco_box_to_bbox(self, box):
        """
        :param box:   输入coco box[x1,y1,w,h]
        :return:      返回bbox[x1,y1,x2,y2]
        """
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __getitem__(self, index):
        #       ＃1。从文件中读取一个数据(例如，使用numpy.fromfile，PIL.Image.open)。
        #        ＃2。预处理数据(例如torchvision.Transform)。
        #        ＃3。返回数据对(例如图像和标签)。
        img_id = self.images[index]  # images 为图片索引
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)  # 读取图片

        height, width = img.shape[0], img.shape[1]  # 高宽
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # 图片中心点
        s = max(height, width) * 1.0  # 高宽中的最大值
        rot = 0

        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = get_border(128, img.shape[1])
            h_border = get_border(128, img.shape[0])
            # 变换中心位置
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if np.random.random() < self.opt.aug_rot:  # 旋转增强的概率,目前default为0 默认不进行旋转增强
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:  # 水平翻转增强的概率 0.5
                flipped = True
                img = img[:, ::-1, :]  # 水平翻转
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:  # 颜色增强
            color_aug(self._data_rng, inp, self.eig_val, self.eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # 变换维度，将通道放在宽和高之前

        output_res = self.opt.output_res  # 输出resolution 为128
        num_joints = self.num_joints  # 关节点数
        # output 变换矩阵
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        # 初始化hm hm_hp kps wh
        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)  # 1*128*128
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)  # 17*128*128

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # max * 2
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)  # max * 34
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # max *2
        ind = np.zeros((self.max_objs), dtype=np.int64)  # index 一维矩阵
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # index 一维矩阵
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)  # mask max,34
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)  # max * 17 , 2
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)  #
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        # ground Truth detection
        gt_det = []
        for k in range(num_objs):  # 对图中物体个数进行循环
            ann = anns[k]  # 读取标签
            bbox = self._coco_box_to_bbox(ann['bbox'])  # 转换成bbox
            cls_id = int(ann['category_id']) - 1  # cls_id = 0
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints,3)  
            # pts = [17,3] 分别为 x y 和visible (0为没有 1为有但不可见，2为可见)
            num_kpts = int(ann['num_keypoints'])  # 读取关键点的数量
            # 如果图像翻转了 就需要对包围框也翻转
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1  # width 为图像的宽
                pts[:, 0] = width - pts[:, 0] - 1  # 对人体关键点的x坐标也进行翻转
                for e in self.flip_idx:  # 对每个对称的点
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()  # 对称点翻转
            # 对包围框也做与图片相同的仿射变换
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)  # 将包围框的坐标限制在(0，127)之间
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # 包围框的 h和w
            if (h > 0 and w > 0) or (rot != 0):
                radius = max(0, gaussian_radius((math.ceil(h), math.ceil(w))))  # 高斯半径
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # 包围框中心
                ct_int = ct.astype(np.int32)  # 坐标转换为int
                wh[k] = 1. * w, 1. * h  # 保存包围框的宽和高
                ind[k] = ct_int[1] * output_res + ct_int[0]  # 保存当前中心坐标在拉平后的索引=y *128 + x
                reg[k] = ct - ct_int  # 偏移量
                reg_mask[k] = 1  # 每个包围框的reg_mask = 1

                # 如果关键点个数为0
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999  # hm[0,y,x] = 0.9999
                    reg_mask[k] = 0  # reg_mask[k] = 0 说明没有偏移量

                hp_radius = radius
                for j in range(num_joints):
                    if pts[j, 2] > 0:  # 对关键点也进行仿射变换
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int  # kps为关键点到中心点的偏移量 shape为[k,34]
                            kps_mask[k, j * 2: j * 2 + 2] = 1  # mask 表示当前位置有无关键点
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int  # 关键点偏移量
                            hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            # if self.opt.dense_hp:   # 是否根据距离中心点远近分配权重
                            #   # must be before draw center hm gaussian
                            #   draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                            #                  pts[j, :2] - ct_int, radius, is_offset=True)
                            #   draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)  # 为每个关节点画高斯圆
                draw_gaussian(hm[cls_id], ct_int, radius)
                """
                # gt_det = [x1,y1,x2,y2,1,pts(17*2),cls_id] 其中1为scores
                """
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0
        """
        ret字典保存了数据集处理的结果
        input     :    仿射变换后的数据
        hm        :    类别的heatmap                                             shape为[1，128，128]
        reg_mask  :    代表当前框有无关键点 ，1代表有，0代表无                        shape为[num_objs]
        ind       :    当前中心点在flatten后的索引                                 shape为[num_objs]
        wh        :    包围框的宽和高                                             shape为[num_objs,2]
        hps       :    所有关键点相对中心点的x y偏移量,排列方式为 x y                 shape为[num_objs,34] 
        hps_mask  :    该位置有无偏移量，有为1，无为0                                shape为[num_objs,34]
        reg       :    中心点的左边偏移量(取整后抹掉的小数部分)                        shape为[num_objs,2]
        hm_hp     :    人体关键点的heatmap                                        shape为[17,128,128]
        hp_offset :    关键点的偏移量                                              shape为[17*num_objs,2]   
        hp_ind    :    当前关键点的位置索引                                         shape为[17*num_objs]
        hp_mask   :    当前位置有无关键点                                           shape为[17*num_objs]
        """
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask, 'reg': reg, 'hm_hp': hm_hp,
               'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask}

        if not self.split == 'train' or self.opt.debug>0:  # 如果是测试集 则输出可视化结果
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            """
            meta 存放的是检测框 只有在测试集和验证集的时候才会使用
            c       ：         中心点坐标 shape为 【2】
            s       ：         s为高和宽中的最大值 int
            gt_det  ：         ground truth检测框
            img_id  :          图片的id
            """
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret

    # COCO EVAL相关

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out  = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
                        np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
                    keypoints  = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))


    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats