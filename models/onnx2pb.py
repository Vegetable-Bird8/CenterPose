from onnx_tf.backend import prepare
import onnx
# import tensorflow as tf
# if not tf.__version__.startswith('1'):
#   import tensorflow.compat.v1 as tf
# from tensorflow.python.tools import optimize_for_inference_lib
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # 去掉警告
TF_PATH = "./dynamic_batch_size/mobilenet_v2_tfmodel" # where the representation of tensorflow model will be stored
ONNX_PATH = "/home/hsn/CenterNet/CenterNet-master/models/dynamic_batch_size/mobilenet_v2.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)


