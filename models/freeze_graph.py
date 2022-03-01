import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
import numpy as np
def frozen_keras_graph(func_model):
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func_model)

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.resource
    ]
    output_tensors = frozen_func.outputs
    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph)

    return graph_def


# def convert_keras_model_to_pb():

#     keras_model = train_model()
#     func_model = tf.function(keras_model).get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
#     graph_def = frozen_keras_graph(func_model)
#     tf.io.write_graph(graph_def, '/tmp/tf_model3', 'frozen_graph.pb')

def convert_saved_model_to_pb():
    model_dir = './mobilenet_v2_tfmodel'
    model = tf.saved_model.load(model_dir)
    func_model = model.signatures["serving_default"]  # signatures默认都为serving_default,保存了输入维度以及输出的维度和名字 
    graph_def = frozen_keras_graph(func_model)
    tf.io.write_graph(graph_def, './tf_model3', 'frozen_graph.pb')


# convert_saved_model_to_pb()
def convert_saved_model_to_pb(output_node_names, input_saved_model_dir, output_graph_dir):
    from tensorflow.python.tools import freeze_graph

    output_node_names = ','.join(output_node_names)

    freeze_graph.freeze_graph(input_graph=None, input_saver=None,
                              input_binary=None,
                              input_checkpoint=None,
                              output_node_names=output_node_names,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=output_graph_dir,
                              clear_devices=None,
                              initializer_nodes=None,
                              input_saved_model_dir=input_saved_model_dir)


def save_output_tensor_to_pb():
    output_names = ['hm','hm_hp','wh','reg','hps','hp_offset']
    save_pb_model_path = './tf_model3/freeze_graph.pb'
    model_dir = './mobilenet_v2_tfmodel'
    convert_saved_model_to_pb(output_names, model_dir, save_pb_model_path)
# save_output_tensor_to_pb()
model_dir = './dynamic_batch_size/mobilenet_v2_tfmodel'
model = tf.saved_model.load(model_dir) 
print(list(model.signatures.keys()))
infer = model.signatures["serving_default"]
# print(infer.inputs)
print("====================>")
# print(infer.structured_outputs)
print("====================>")
print(infer.outputs)

# frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(infer)

# input_tensors = [
#     tensor for tensor in frozen_func.inputs
#     if tensor.dtype != tf.resource
# ]
# print('==================================================================>\n',input_tensors)
# output_tensors = frozen_func.structured_outputs
# graph_def = run_graph_optimizations(
#     graph_def,
#     input_tensors,
#     output_tensors,
#     config=get_grappler_config(["constfold", "function"]),
#     graph=frozen_func.graph)
# tf.io.write_graph(graph_def, './tf_model4', 'frozen_graph.pb')
# loaded = tf.saved_model.load(model_dir) 
# print('MobileNet has {} trainable variables: {},...'.format(
#        len(loaded.trainable_variables),
#        ', '.join([v.name for v in loaded.trainable_variables[:5]])))
# trainable_variable_ids = {id(v) for v in loaded.trainable_variables}
# non_trainable_variables = [v for v in loaded.variables if id(v) not in trainable_variable_ids]
# print('MobileNet also has {} non-trainable variables: {}, ...'.format(
#        len(non_trainable_variables),
#        ', '.join([v.name for v in non_trainable_variables[:3]])))
output_names = ['Identity:0',"Identity_1:0",'Identity_2:0','Identity_3:0','Identity_4:0','Identity_5:0']
output_names = ['PartitionedCall:0','PartitionedCall:1','PartitionedCall:2','PartitionedCall:3','PartitionedCall:4','PartitionedCall:5']
# # output_names = ['hm','hm_hp','wh','reg','hps','hp_offset']
output_node_names = ','.join(output_names)
print(output_node_names)
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='/home/hsn/CenterNet/CenterNet-master/models/Test_(1,3,512,512)/saved_model/mobilenet_v2_tfmodel', input_saver='',
                            input_binary=None,
                            input_checkpoint=None,
                            output_node_names=output_node_names,
                            restore_op_name='save/restore_all',
                            filename_tensor_name=None,
                            output_graph='./Test_(1,3,512,512)/saved_model/frozen_graph.pb',
                            clear_devices=None,
                            initializer_nodes=None,
                            input_saved_model_dir='/home/hsn/CenterNet/CenterNet-master/models/Test_(1,3,512,512)/saved_model/mobilenet_v2_tfmodel')