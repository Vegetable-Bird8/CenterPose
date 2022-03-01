# 查看saved_model文件的input和output
```
python saved_model_cli.py show --dir /home/hsn/CenterNet/CenterNet-master/models/mobilenet_v2_tfmodel --all
"""
输出信息
"""
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3, 512, 512)
        name: serving_default_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['hm'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 1, 128, 128)
        name: PartitionedCall:0
    outputs['hm_hp'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 17, 128, 128)
        name: PartitionedCall:1
    outputs['hp_offset'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 2, 128, 128)
        name: PartitionedCall:2
    outputs['hps'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 34, 128, 128)
        name: PartitionedCall:3
    outputs['reg'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 2, 128, 128)
        name: PartitionedCall:4
    outputs['wh'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 2, 128, 128)
        name: PartitionedCall:5
  Method name is: tensorflow/serving/predict


```