import os.path as osp
from warnings import simplefilter
import numpy as np
import onnx
import onnxruntime as ort
import torch
# import torchvision
import time
import sys
# import onnx-simplefilter as onnxsim
sys.path.append(osp.join(osp.dirname(__file__), './'))
sys.path.append(osp.join(osp.dirname(__file__), '../'))
from src.lib.models.networks.MobileNet_v2 import get_MobileNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# torch --> onnx

test_arr = np.random.randn(1, 3, 512, 512).astype(np.float32)
# print(type(test_arr))
# test_arr = torch.from_numpy(test_arr)
# print(type(test_arr),test_arr.shape)
dummy_input = torch.tensor(test_arr)
# print(dummy_input.shape)
# model = torchvision.models.mobilenet_v2(pretrained=True).eval()
# gpu 
# model = get_MobileNet(train_or_test='test').eval().to('cuda')
model = get_MobileNet(train_or_test='test').eval()
check_point = torch.load('/home/hsn/CenterNet/CenterNet-master/exp/multi_pose/default/mobilenet_v2_train2/model_119.pth',map_location=torch.device('cpu'))
# print(check_point.keys())
model.load_state_dict(check_point['state_dict'])
start = time.time()
# torch_output = model(torch.from_numpy(test_arr).to('cuda'))
torch_output = model(torch.from_numpy(test_arr))
print("cpu下网络的时间为：",time.time()-start)
# print(f"torch输出的长度为：{len(torch_output)},类型为：{type(torch_output)},\n，内部类型为：{type(torch_output[-1])},长度为：{len(torch_output[-1])}")

# 以下为转换为onnx
# 在此示例中，我们使用输入batch_size=1导出模型，
# 但随后在torch.onnx.export()的dynamic_axes参数中将第一维指定为动态。 
# 因此，导出的模型将接受大小为[batch_size,3, 640, 959]的输入，其中batch_size可以是可变的。
input_names = ["input"]
output_names = ['hm','hm_hp','wh','reg','hps','hp_offset']
# torch.onnx.export(model, 
#                   dummy_input, 
#                   "./models/mobilenet_v2.onnx", 
#                   verbose=False, 
#                   input_names=input_names, 
#                   output_names=output_names,
#                   opset_version=12,
#                   do_constant_folding=True,
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                 'hm': {0 : 'batch_size'},'hm_hp': {0 : 'batch_size'},
#                 'wh': {0 : 'batch_size'},'reg': {0 : 'batch_size'},
#                 'hps': {0 : 'batch_size'},'hp_offset': {0 : 'batch_size'}})

# torch.onnx.export(model,               # model being run
#                 dummy_input,                         # model input (or a tuple for multiple inputs)
#                 "./models/mobilenet_v2.onnx",   # where to save the model (can be a file or file-like object)
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=12,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'], # the model's output names
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}})
"""
ONNX模型主要保存了五个结构，通过load网络模型，打印graph结构，可以清晰的查看存储的内容。

node：网络节点列表,

initializer：所有常量输入数据列表

,input：网络中存在的feature map输入列表,

output：整个模型的输出列表。
"""

# 转换后加载模型
model = onnx.load("./model/dynamic_batch_size/mobilenet_v2.onnx")
# graph = model.graph
# print(f"模型的结构为{graph}")
# # 图结构的输出
# print(f"graph的输出为:{graph.output}")

# #print intermediate node info
# print(graph.node[37])
# print(graph.node[37].input[0])
# 创建runtime实例
sess = ort.InferenceSession('./model/dynamic_batch_size/mobilenet_v2.onnx')
"""
# 其中input 和 output 都为list类型，
# input name 为input512
# output存放了6个输出头，和原版一致。
# output name 分别为  ['hm','hm_hp','wh','reg','hps','hp_offset']
# print(type(sess))"""
# input_name = sess.get_inputs()[0].shape  #list 长度为1 
# print("input name", input_name,'\n',len(input_name))
# output_name= sess.get_outputs() # list 长度为6
# print("output name\n", len(output_name),"\n",type(output_name))
# output_shape = sess.get_outputs()[-1].name
# print("output shape\n", output_shape,"\n",type(output_shape))

# # 
start = time.time()
onnx_outputs = sess.run(output_names=output_names, input_feed={'input': test_arr})
print("cpu下onnx网络的推理时间为：",time.time()-start)
# print(f"onnx 输出的长度为：{len(onnx_outputs)}")

print('=====================================>Export ONNX!')

torch_outputs = [torch_output[-1][x] for x in torch_output[-1]]
# for i in range 
# print(f"torch输出的长度为：{len(torch_outputs)},类型为：{type(torch_outputs)},\n，内部类型为：{type(torch_outputs[-1])},长度为：{len(torch_outputs[-1])}")
dif = []
for i in range(len(torch_outputs)):
    torch_outputs[i] = torch_outputs[i].cpu().detach().numpy()
    # print(torch_outputs[i],"\n====================\n",onnx_outputs[i])
    # print("===========================>")
    dif.append(pow(torch_outputs[i] - onnx_outputs[i],2).mean())
print("L2偏差为：",sum(dif),dif)
# cpu下网络的时间为： 0.1518232822418213
# cpu下onnx网络的推理时间为： 0.054608821868896484
# =====================================>Export ONNX!
# L2偏差为： 0.0001896952983315714 [3.2159278e-08, 1.3027227e-06, 5.383106e-05, 0.0, 0.00013452936, 0.0]
# print("====> Simplifying...")
# model_opt = onnxsim.simplify(args.onnx_model)
# # print("model_opt", model_opt)
# onnx.save(model_opt, args.onnx_model_sim)
# print("onnx model simplify Ok!")