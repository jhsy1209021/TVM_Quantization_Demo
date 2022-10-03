import tensorflow_model_optimization as tfmot
import tensorflow as tf
import keras
import tvm.relay as relay
import numpy as np
import keras.datasets.mnist as mnist

# Data preprocess
## Do normalization
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test = x_test/255
## Flatten the array
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
#data = x_test[2]
#data = gray.reshape(28,28,1)
#data = data/255
#np.testing.assert_allclose(data_test,data,atol=1e-1)
#data = data.reshape(1, 28, 28, 1)
#data = np.transpose(data, (0, 3, 1, 2))

#Annotate the layer that we want to quantize
MODEL_PATH  = "../model/lenet5/lenet5_model.h5"
WEIGHT_PATH = "../model/lenet5/lenet5_weight.h5"
keras_lenet5 = keras.models.load_model(MODEL_PATH)
keras_lenet5.load_weights(WEIGHT_PATH)
keras_lenet5.summary()

quant_list = ['dense_1', 'dense_2']
def apply_quantization_to_dense(layer):
  if (layer.name in quant_list):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

annotated_model = tf.keras.models.clone_model(
    keras_lenet5,
    clone_function=apply_quantization_to_dense,
)
annotated_model.load_weights(WEIGHT_PATH)

quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_aware_model.summary()
#quant_aware_model.save('./layerbylayer_quant.h5')

#get full quatize model by tensorflow lite converter
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 28, 28, 1)
      yield [data.astype(np.float32)]

"""
converter = tf.lite.TFLiteConverter.from_keras_model(keras_lenet5)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
lenet5_prequant = converter.convert()
"""

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
LayerbyLayer = converter.convert()
with open("../model/lenet5/Lenet5_layerbylayer_quant" + ".tflite", 'wb') as f:
        f.write(LayerbyLayer)


##########Layer Bt Layer Qnatization By TVM##########
shape_dict = {"conv2d_1_input": (1, 1, 28, 28)}
mod, params = relay.frontend.from_keras(keras_lenet5, shape_dict)


with relay.quantize.qconfig(    calibrate_mode="global_scale",
                                global_scale=8.0,
                                nbit_activation=16,
                                dtype_activation="int16",
                                skip_conv_layers=[],
                                skip_dense_layer=True,
                                partition_conversions="disabled"):
    mod = relay.quantize.quantize(mod, params)
    #viz = relay_viz.RelayVisualizer(mod)
    #viz.render()
    # Relay graph after quantization
    with open("Lenet_LayerbyLayer_quant_tvm.log",'w') as f:
        k_relay = mod.astext(show_meta_data = False)
        f.write(k_relay)
