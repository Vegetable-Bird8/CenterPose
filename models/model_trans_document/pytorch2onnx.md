# Pth模型转tflite并量化



## Step 1 pytorch2onnx

https://www.cnblogs.com/happyamyhope/p/14838292.html

```
import os.path as osp
from warnings import simplefilter
import numpy as np
import onnx
import onnxruntime as ort
import torch
# import torchvision  
# 若需要获得现成的模型 导入torchvision
import sys
# 简化onnx模型需要的工具
import onnx-simplefilter as onnxsim
#  下列用于将上级目录加入工作空间
#  方便导入代码
sys.path.append(osp.join(osp.dirname(__file__), './'))
sys.path.append(osp.join(osp.dirname(__file__), '../'))
from src.lib.models.networks.MobileNet_v2 import get_MobileNet

# 测试数据 batch_size一般设置为1
# 如果不设置batch_size为动态轴 一般设置为1 
test_arr = np.random.randn(1, 3, 512, 512).astype(np.float32)

# test_arr = torch.from_numpy(test_arr)

dummy_input = torch.tensor(test_arr)
# 以下通过torchvision直接获得网络模型（带有预训练模型）
# model = torchvision.models.mobilenet_v2(pretrained=True).eval()
# 自定义网络
model = get_MobileNet(train_or_test='test').eval()
# 加载权重文件，格式为字典
check_point = torch.load('/home/hsn/CenterNet/CenterNet-master/exp/multi_pose/default/mobilenet_v2_train2/model_119.pth',map_location=torch.device('cpu'))
# print(check_point.keys())
# 将权重加载入模型中
model.load_state_dict(check_point['state_dict'])
start = time.time()
torch_output = model(torch.from_numpy(test_arr))

# 以下为转换为onnx
input_names = ["input"]  #输入的name
output_names = ['hm','hm_hp','wh','reg','hps','hp_offset'] #输出的name
torch.onnx.export(model, 
                  dummy_input, 
                  "./models/mobilenet_v2.onnx", 
                  verbose=False, 
                  input_names=input_names, 
                  output_names=output_names)
                  
torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnxpath,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                    'output' : {0 : 'batch_size'}})
"""
ONNX模型主要保存了五个结构，通过load网络模型,可以分辨看出五大部分内容
graph:打印graph结构，可以清晰的查看存储的内容。

node：网络节点列表,

initializer：所有常量输入数据列表

input：网络中存在的feature map输入列表,

output：整个模型的输出列表。
"""

# 转换后加载onnx模型
model = onnx.load("./models/mobilenet_v2.onnx")
# graph = model.graph
# print(f"模型的结构为{graph}")
# # 图结构的输出
# print(f"graph的输出为:{graph.output}")
# # 打印中间节点
# print(graph.node[37])
# print(graph.node[37].input[0])

# 创建runtime实例
sess = ort.InferenceSession('./models/mobilenet_v2.onnx')
# 输入得到结果
onnx_outputs = sess.run(output_names=output_names, input_feed={'input': test_arr})
#输出的结果应与pytorch的一致
```

## Step 2 onnx simplifier

onnx模型去掉冗余结构，有利于加快模型速度

```
python -m onnxsim --skip-optimizer fuse_consecutive_concats  your.onnx your_simplified.onnx
```



## Step 3 onnx2pb（saved_model）

```
from onnx_tf.backend import prepare
import onnx

TF_PATH = "sim_tf_model" # where the representation of tensorflow model will be stored
ONNX_PATH = "sim_v2.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)
```

## Step4  pb2tflite and Quantization

```\
import tensorflow as tf

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1,3,512,512)
        yield [data.astype(np.float32)]
# 生成代表性数据集
def get_representative_dataset():
    img_dir = r"数据"
    img_ext = ".bmp"
    img_paths = glob.glob(os.path.join(img_dir, '*' + img_ext))
    for path in img_paths:
        img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),0)
        img= cv2.resize(img, (512,512), interpolation = cv2.INTER_NEAREST)/255
        img=img.reshape((1,512,512,1)).astype(np.float32)
        yield  img,img

TF_PATH = "./dynamic_batch_size/mobilenet_v2_tfmodel" 
TFLITE_PATH = "./dynamic_batch_size/mobilenet_v2.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter =True
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]

converter.inference_type = tf.uint8    #tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)} # mean, std_dev
converter.default_ranges_stats = (0, 255)
# converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS
# ]
converter.inference_input_type = tf.int8 
converter.inference_
converter.inference_output_type = tf.int8
converter.allow_custom_ops=True

# tflite_model = converter.convert()
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
```

