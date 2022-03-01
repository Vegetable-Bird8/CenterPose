import tensorflow as tf
import numpy as np
import glob,os,cv2
 
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


#----------------加载并获取基本信息---------------------
interpreter = tf.lite.Interpreter(model_path='/home/hsn/CenterNet/CenterNet-master/models/dynamic_batch_size/mobilenet_v2.tflite')
#interpreter = tf.lite.Interpreter(model_content=concrete_func_tflite)
interpreter.allocate_tensors()#给所有的tensor分配内存
 
#获取 input 和 output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
#输出字典，我们需要利用里面的输出来设置输入输出的相关参数
print(input_details)
print("================>")
print(output_details)
 
# #----------------进行预测---------------------
# input_shape = input_details[0]['shape']#设置输入维度
data_gen=get_representative_dataset()
# x,x=data_gen.__next__()
# input_data = tf.constant(x)
# interpreter.set_tensor(input_details[0]['index'],input_data)
 
# #执行预测
# interpreter.invoke()
# output_results = interpreter.get_tensor(output_details[0]['index'])
# print(output_results.shape)#十个标签概率值的概率分布
# import matplotlib.pyplot as plt
# plt.imshow(output_results[0,:,:,0])
# plt.show()
# plt.imshow(x[0,:,:,0])
# plt.show()
# 因为上一步保存的模型文件已经是pb格式了，所以不用先转为pb，如果不是pb格式，参考：https://blog.csdn.net/qxqxqzzz/article/details/119668426?spm=1001.2014.3001.5501
def tf_tflite():
    tf_model_path, tflite_model_path = './tfmodel', 'model.tflite'
    converter = tf.lite.TFLiteCOnverter.from_saved_model(tf_model_path)
    converter.target_spec,supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINGS,tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as g:
        g.write(tflite_model)

def tflite_prediction(img_batch):
    tflite_model = 'model.tflite'
    interpreter = tf.lite.Interpreter(model_path = tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter,set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index']) # output feature
    tflite_pred = tf.convert_to_tensor(tflite_pred)
    tflite_pred = tf.nn.softmax(tflite_pred)
    print(tf.argmax(tflite_pred, 1)) # output class index
