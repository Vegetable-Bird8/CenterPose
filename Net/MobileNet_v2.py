from pickle import TRUE
import torch
from torch import nn
BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class BasicBlock(nn.Module):
# #   expansion = 1

#   def __init__(self, inplanes, planes, stride=1, downsample=None):
#     super(BasicBlock, self).__init__()
#     self.conv1 = conv3x3(inplanes, planes, stride)
#     self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#     self.relu = nn.ReLU(inplace=True)
#     self.conv2 = conv3x3(planes, planes)
#     self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#     self.downsample = downsample
#     self.stride = stride

#   def forward(self, x):
#     residual = x

#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu(out)

#     out = self.conv2(out)
#     out = self.bn2(out)

#     if self.downsample is not None:  #如果降采样了 
#       residual = self.downsample(x)

#     out += residual
#     out = self.relu(out)

#     return out

"""
mobilenet v2 的revert residual block
"""
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):  #t 为扩张倍数 默认为6
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
            nn.BatchNorm2d(in_channels * t,momentum= BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t,
                      bias=False),
            nn.BatchNorm2d(in_channels * t,momentum= BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels,momentum= BN_MOMENTUM)
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        if self.stride == 1:
            out += self.shortcut(x)
        return out
"""
MobildNetv2 构成的backbone 输入默认为512*512*3
提取特征，最终获得 16*16*2048的特征层
"""
class MobileNet(nn.Module):
    def __init__(self):
        self.feature_channels =2048
        self.deconv_with_bias = False
        self.num_classes = 1
        self.num_joints = 17
        super().__init__()

        """
        1.
        BackBone网络 接收输入的图片
        经过Mobilenet_v2的网络结构
        输出2048*16*16的深层特征
        """
        self.conv1 = nn.Sequential(   # 第一个卷积层 输出通道32  STRIDE = 2
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32,momentum=0.9),
            nn.ReLU6(inplace=True)
        )  # 512/2 =256
        self.bottleneck1 = self._make_layer(1, 32, 16, 1, 1)
        self.bottleneck2 = self._make_layer(2, 16, 24, 2, 6)  #256/2=128
        self.bottleneck3 = self._make_layer(3, 24, 32, 2, 6)  #128/2=64
        self.bottleneck4 = self._make_layer(4, 32, 64, 2, 6)  #64/2=32
        self.bottleneck5 = self._make_layer(3, 64, 96, 1, 6)
        self.bottleneck6 = self._make_layer(3, 96, 160, 2, 6)  #32/2=16
        self.bottleneck7 = self._make_layer(1, 160, 320, 1, 6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(320,2048,kernel_size = 1,stride=1,bias=False),
            nn.BatchNorm2d(2048,momentum=0.9),
            nn.ReLU6(inplace=True)
        ) # 修改通道数
        """
        2.
        上采样网络 将特征编码器输出的 16*16*2048特征进行解码
        经过三层上采样，将特征图变为 128*128*64的特征图
        为下面的分支网络提供输入
        """
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,  # 上采样次数
            up_sampling_channels=[256, 128, 64],  # 三次上采样 每次输出的通道数
            kernel_sizes=[4, 4, 4],  # 卷积核尺寸
        )
        """
        3.
        经过上采样后 特征变为128*128*64的特征图
        CenterPose共有六个分支的输出：（H=W=128)
            Center Heatmap:1×H×W，类别heatmap中心点预测，最大值即为目标中心，最大值对应坐标为中心坐标
            Center offset: 2×H×W, 预测中心与真值的偏移量
            Center Box size: 2×H×W，每个目标中心矩形的宽和高 
            Center Points coordination:34×H×W，关节坐标相对于中心坐标的偏移
            ----------------------------------------------------------------
            Points Heatmap: 17×H×W, 关节的HeatMap 17个关节点
            Points offset: 2×H×W，关节点的坐标偏移量 
            一共预测了两套坐标，上面四个分支为一套坐标，下面单独为17个坐标点预测坐标和偏移
            偏移的出现主要在于降采样和升采样时产生的偏移，因为所有点采样策略都相同，因此默认偏移量都相同
        """
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, 1
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 34
        #                -> 128, 128, 64 -> 128, 128, 17
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # 类别热力图预测部分
        self.hm = nn.Sequential(
            nn.Conv2d(64, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1,
                      kernel_size=1, stride=1, padding=0))

        # 宽高预测的部分
        self.wh = nn.Sequential(
            nn.Conv2d(64, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的偏移量
        self.reg = nn.Sequential(
            nn.Conv2d(64, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1, padding=0))

        # 预测人体关键点相对于中心点的偏移量
        self.hps = nn.Sequential(
            nn.Conv2d(64,64,
                      kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,self.num_joints*2,
                      kernel_size=1,stride=1,padding = 0)
        )

        # 人体关键点热力图预测部分
        self.hm_hp = nn.Sequential(
            nn.Conv2d(64,64,
                      kernel_size = 3,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,self.num_joints,
                      kernel_size=1,stride=1,padding=0)
        )

        # 人体关键点偏移量
        self.hp_offset = nn.Sequential(
            nn.Conv2d(64, 64,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1, padding=0))
        """
        mobilenet 的 reverse Resnet Block的叠加 
        每个layer 带有一次降采样
        """
    def _make_layer(self, repeat, in_channels, out_channels, stride, t):   # 用来控制block数量 t为通道扩张倍数
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, t))  # 只有第一层进行降采样 in ！=out
        while repeat-1:
            layers.append(BottleNeck(out_channels, out_channels, 1, t)) # 其余层stride都为1
            repeat -= 1
        return nn.Sequential(*layers)  # 必须用*将list转化


        """     
            16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
            利用ConvTranspose2d进行上采样
            每次特征层的宽高变为原来的两倍
        """
    def _make_deconv_layer(self, num_layers, up_sampling_channels, kernel_sizes):
        layers = []
        for i in range(num_layers):
            channels = up_sampling_channels[i]
            kernel_size = kernel_sizes[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.feature_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(channels,momentum=0.9))
            layers.append(nn.ReLU(inplace=True))
            self.feature_channels = channels  # 将下一层卷积的inchannels 换成上一层卷积的outchannels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.conv2(x)
        x = self.deconv_layers(x)
        output={}
        output['hm'] = self.hm(x)
        output['hm_hp'] = self.hm_hp(x)
        output['wh'] = self.wh(x)
        output['reg'] = self.reg(x)
        output['hps'] = self.hps(x)
        output['hp_offset'] = self.hp_offset(x)
        return [output]


    # #使用官方代码提供的方法：
    def special_init_weights(self):
        #初始化转置卷积和BN
        # for _, m in self.deconv_layers.named_modules():  # name,module
        # for m in self.deconv_layers.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         if self.deconv_with_bias:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
        for m in self.hm.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.hm_hp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -2.19)
        for m in self.wh.modules():
            if isinstance(m, nn.Conv2d):    
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)      
        for m in self.reg.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.hps.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.hp_offset.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # #初始化一下三个输出层'
        # heads = {'hm' : 1,'hm_hp':17,'wh':2,'reg':2,'hps':34,'hp_offset':2}
        # for head in heads:
        #     final_layer = self.__getattr__(head)
        #     print(final_layer)
        #     for i, m in enumerate(final_layer.modules()):
        #         print(i,'------',m)
        #         if isinstance(m, nn.Conv2d):
        #             if m.weight.shape[0] == heads[head]:
        #                 if 'hm' in head:
        #                     nn.init.constant_(m.bias, -2.19)
        #                 else:
        #                     nn.init.normal_(m.weight, std=0.001)
        #                     nn.init.constant_(m.bias, 0)

            # #加载resnet预训练权重
            # url = 
            # pretrained_state_dict = model_zoo.load_url(url)
            
            # print('\n=> loading pretrained model {}'.format(url))
            # self.load_state_dict(pretrained_state_dict, strict=False)


"""    
    'hm': 1,  # 6       类别中心点预测

    'wh': 2,  # 11    矩形的宽和高 
    'offset': 2,  # 10   包围框的坐标偏移
    'hps': 34,  # 9   各关键点相对于中心点的坐标偏移（输出时应加上中心点坐标）
    'hp_hm': 17,  # 7    human pose heat map关键点的预测
    'hp_offset': 2,  # 8  关键点的偏移


"""
def all_weight_init(m):
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.001)

        

def get_MobileNet( pretrained_file='',train_or_test='train'):
    model = MobileNet()
    if train_or_test == 'train':
        model.apply(all_weight_init)
        model.special_init_weights()
    # print(model)
    return model

if __name__ == '__main__':
  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)


  net = get_MobileNet()
  print(net)
#   load_model = 


#   net = get_hourglass['large_hourglass']
#   load_model(net, '../ckpt/pretrain/checkpoint.t7')
#   count_parameters(net)
#   count_flops(net, input_size=512)

#   for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#       m.register_forward_hook(hook)

#   with torch.no_grad():
#     y = net(torch.randn(2, 3, 512, 512).cuda())

  y = net(torch.randn(2, 3, 512, 512))
  # y.summary()
  print(type(y[0]),len(y),y[0]['hps'].shape,type(y))
  print(y[-1]['wh'][0:1].shape)