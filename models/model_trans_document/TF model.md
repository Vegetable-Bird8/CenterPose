# TF三种模型

## 1 weight(.ckpt)

只保存网络的一个参数，不管其他的状态，这种模式适合自己对代码有个清晰的认识

```
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')  # 提供保存的路径

# Restore the weights
model = create_model()  # 重新创建网络
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)  # 查看accuracy是否变化
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

```

用法示例：

```
network.save_weights('weights.ckpt')
print('saved weights.')
del network

network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
network.load_weights('weights.ckpt')
network.evaluate(ds_val)

```

## 2 Net & Weight(.h5)

把所有的模型和状态都保存起来，可以进行完美的恢复，主要来自于keras的默认模型格式

```
network.save('model.h5')
print('saved total model.')
del network # # del删除的是变量 不是数据

print('loaded model from file.')
network = tf.keras.models.load_model('model.h5', compile=False)  # 不需要重新创建网络

network.evaluate(ds_val)

```

## 3 saved_model

模型的一种保存格式，跟pytorch的ONNX对应

也就是说当训练的一个模型交给工厂的生产环境的时候，可以把这个模型直接交给用户来部署，而不需要给一个源代码或相关的信息，这个模型就包含的所有的这样一个信息。

比如，你通过python写的源文件，你可以用c++解析和读取这个工作。

```
tf.saved_model.save(m, '/tmp/saved_model')

imported = tf.saved_model.load(path)
f = imported.signatures["serving_default"]
print(f(input = tf.ones([1, 28, 28, 3])))

```

