import tensorflow as tf
import numpy as np
# from src.lib.models.networks.MobileNet_v2 import get_MobileNet

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1,3,512,512)
        yield [data.astype(np.float32)]
        
# model = HourglassNetwork(heads=heads, **kwargs)
# model = get_MobileNet(train_or_test='test')
# # print(model.outputs[3])
# # model = HpDetDecode(model)
# model.summary()

# x = [-1, 0, 1, 2, 3, 4]
# y = [-3, -1, 1, 3, 5, 7]

# model = tf.keras.models.Sequential(
#     [tf.keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
# model.fit(x, y, epochs=50)
# model.summary()
# # 8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter =True
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
# # converter.target_spec.supported_ops = [
# #         tf.lite.OpsSet.TFLITE_BUILTINS
# # ]
# converter.inference_input_type = tf.int8 
# converter.inference_output_type = tf.int8
# converter.allow_custom_ops=True

# tflite_model = converter.convert()

# with open("Test.tflite", "wb") as f:
#     f.write(tflite_model)

# import tensorflow as tf

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