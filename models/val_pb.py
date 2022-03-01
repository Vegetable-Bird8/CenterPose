# from torch.functional import Tensor
import tensorflow as tf
import onnx
# import os.path as osp
from warnings import simplefilter
import numpy as np
import onnxruntime as ort

import time
import sys
# if not tf.__version__.startswith('1'):
#   import tensorflow.compat.v1 as tf
# from tensorflow.python.tools import optimize_for_inference_lib

# graph_def_file = "./Test_(1,3,512,512)/tf_model4/frozen_graph.pb"

# tf.reset_default_graph()
# graph_def = tf.GraphDef()
# with tf.Session() as sess:
#     # Read binary pb graph from file
#     with tf.gfile.Open(graph_def_file, "rb") as f:
#         data2read = f.read()
#         graph_def.ParseFromString(data2read)
#     tf.graph_util.import_graph_def(graph_def, name='')
    
#     # Get Nodes
#     conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool','ConvTranspose2d']]
#     for n_org in conv_nodes:
#         # Transpose input
#         assert len(n_org.inputs)==1 or len(n_org.inputs)==2
#         org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
#         inp_tens = tf.transpose(org_inp_tens, [0, 2, 3, 1], name=n_org.name +'_transp_input')
#         op_inputs = [inp_tens]
        
#         # Get filters for Conv but don't transpose
#         if n_org.type == 'Conv2D':
#             filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
#             op_inputs.append(filter_tens)
        
#         # Attributes without data_format, NWHC is default
#         atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
#         if n_org.type in['MaxPool', 'AvgPool','Conv2D','ConvTranspose2d']:
#             st = atts['strides'].list.i
#             stl = [st[0], st[2], st[3], st[1]]
#             atts['strides'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))
#         if n_org.type in ['MaxPool', 'AvgPool']:
#             st = atts['ksize'].list.i
#             stl = [st[0], st[2], st[3], st[1]]
#             atts['ksize'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))

#         # Create new Operation
#         #print(n_org.type, n_org.name, list(n_org.inputs), n_org.node_def.attr['data_format'])
#         op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts) 
#         out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
#         out_trans = tf.transpose(out_tens, [0, 3, 1, 2], name=n_org.name +'_transp_out')
#         assert out_trans.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape
        
#         # Update Connections
#         out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
#         for out in out_nodes:
#             for j, nam in enumerate(out.inputs):
#                 if n_org.outputs[0] == nam:
#                     out._update_input(j, out_trans)
        
#     # Delete old nodes
#     graph_def = sess.graph.as_graph_def()
#     for on in conv_nodes:
#         graph_def.node.remove(on.node_def)

#     # Write graph
#     tf.io.write_graph(graph_def, "", graph_def_file.rsplit('.', 1)[0]+'_toco.pb', as_text=False)

# test_arr = np.random.randn(1, 3, 512, 512).astype(np.float32)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2 as cv
img_path = './Test.jpg'

img = cv.imread(img_path)
# cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
# cv.imshow('lena',img)
# cv.waitKey(3000)
print(img.shape)
img = cv.resize(img,(512,512))
print(img.shape)
# cv.namedWindow('lena',cv.WINDOW_AUTOSIZE)
# cv.imshow('lena',img)
# cv.waitKey(3000)
# img = img.resize(512,512)
# print(img.shape)
test_arr = np.float32(img)
print(test_arr.shape)



Tensor_test = tf.convert_to_tensor(test_arr)
Tensor_test = tf.transpose(Tensor_test, [2,0,1])
Tensor_test = tf.expand_dims(Tensor_test,0)
test_arr = Tensor_test.numpy()
# print(Tensor_test.shape)
# print(type(Tensor_test))
# print(type(tf.ones([1,3,512,512])))
path = './model/dynamic_batch_size/mobilenet_v2_tfmodel'
imported = tf.saved_model.load(path)
f = imported.signatures["serving_default"]
# print(f.structured_outputs)
# print("====================>")
# print(f.outputs)

start = time.time()
tf_out = f(input = Tensor_test)
print(f"tf用时：{time.time()-start}")
# print(len(tf_out),'type \n ===================>',type(tf_out['hm']))



output_names = ['hm','hm_hp','wh','reg','hps','hp_offset']
# model = onnx.load("./model/dynamic_batch_size/mobilenet_v2.onnx")
sess = ort.InferenceSession('./model/dynamic_batch_size/mobilenet_v2.onnx')

start = time.time()
# test_arr = np.random.randn(1, 3, 512, 512).astype(np.float32)
onnx_outputs = sess.run(output_names=output_names, input_feed={'input': test_arr})
print(f"onnx用时：{time.time()-start}")
print(len(onnx_outputs),type(onnx_outputs))
tf_out = [tf_out[x] for x in output_names]
print(len(tf_out),type(tf_out))
dif = []
for i in range(len(onnx_outputs)):
    tf_out[i] = tf_out[i].numpy()  # 等于detach
    # print(tf_out[i])
    # print(onnx_outputs[i])
    # print(torch_outputs[i],"\n====================\n",onnx_outputs[i])
    # print("===========================>")
    # print("===========================>")
    # print("===========================>")
    dif.append(pow(tf_out[i] - onnx_outputs[i],2).mean())
print("L2偏差为：",sum(dif),dif)
#
# L2偏差为： 0.0026649485080270097 [0.00014431572, 0.00016316606, 0.0017568816, 0.0, 0.00060058513, 0.0]
