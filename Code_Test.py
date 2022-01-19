from torch import nn
# head = []
# class A(nn.Module):
#     def __init__(self, value):
#         self.value = value
#
#
# a = A(10)
# print(a.value)
# # 10
# print(a.name)
# # into __getattr__
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mse_loss', action='store_true',
                         help='use mse loss or focal loss to Trainer '
                              'keypoint heatmaps.')

opt = parser.parse_args()
# print(opt.mse_loss)
# import numpy as np
# a = [[1,2,3,4,5]]
# b = {1:[1,2,3,4,5]}
# results = {}
# results[1] = np.concatenate(
#     [detection for detection in a], axis=0).astype(np.float32)
# # if self.opt.nms or len(self.opt.test_scales) > 1:
# #   soft_nms_39(results[1], Nt=0.5, method=2)
# results[1] = results[1].tolist()
# print(type(results),type(results[1]),results)

# print(b[1])
# result = {}
# result[1] = np.array(b[1]).astype(np.float32)
# # if self.opt.nms or len(self.opt.test_scales) > 1:
# #   soft_nms_39(results[1], Nt=0.5, method=2)
# result[1] = result[1].tolist()
# print(type(result),type(result[1]),result,results==result)

EDGES = [[0, 1], [0, 2], [1, 3], [2, 4],
         [4, 6], [3, 5], [5, 6],
         [5, 7], [7, 9], [6, 8], [8, 10],
         [6, 12], [5, 11], [11, 12],
         [12, 14], [14, 16], [11, 13], [13, 15]]
edge = [[0, 1], [0, 2], [1, 3], [2, 4],
              [3, 5], [4, 6], [5, 6],
              [5, 7], [7, 9], [6, 8], [8, 10],
              [5, 11], [6, 12], [11, 12],
              [11, 13], [13, 15], [12, 14], [14, 16]]
# print(len(EDGES),len(edge))

# multi_pose = {
# 'default_resolution': [512, 512], 'num_classes': 1, 
# 'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
# 'dataset': 'coco_hp', 'num_joints': 17,
# 'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
#                 [11, 12], [13, 14], [15, 16]]}
# print(multi_pose.flip_idx)
import numpy as np
print(np.random.random())
