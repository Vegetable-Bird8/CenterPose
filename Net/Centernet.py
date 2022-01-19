# # under windows
# # -*- coding: UTF-8 -*-
# '''=================================================
# @Project -> File   ：CenterPose -> decode
# @IDE    ：PyCharm
# @Author ：SN.Han
# @Date   ：2021/12/14 18:22
# @Desc   ：
# =================================================='''

# ToDo.....
# import torch
# from torch import nn
# from Net.MobileNet_v2 import MobileNetv2,MobileNet2_Decoder,Head
#
# class Centernet(nn.Module):
#     def __init__(self,pretrained_file='',num_classes=1,num_joints =17,pretrained = False):
#         super(Centernet,self).__init__()
#         self.backbone = MobileNetv2('',pretrained=False)
#         # backbone 输出16*16*2048的特征图
#         # 输入decoder中
#         self.decoder = MobileNet2_Decoder(2048)
#         # 对特征图进行上采样 恢复到128*128大小 通道数为64
#         # 输入分支网络中
#         self.head = Head()
#     # 以下代码用于冻结参数，适合与Fine-tuning
#     def freeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         for param in self.backbone.parameters():
#             param.requires_grad = True
#
#     def forward(self, x):
#         feat = self.backbone(x)
#         return self.head(self.decoder(feat))
#     #head 形状为[hm, wh, offset,hps,hp_hm,hp_offset]的列表
#
# if __name__ == '__main__':
#   def hook(self, input, output):
#     print(output.data.cpu().numpy().shape)
#
#
#   net = Centernet()
#   print(net)
#
#   for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#       m.register_forward_hook(hook)
#
#   y = net(torch.randn(2, 3, 512, 512))
#   # y.summary()
#   print(type(y[0]),len(y),y[0].shape,type(y))
