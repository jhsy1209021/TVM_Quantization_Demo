import os
import tvm
from tvm import te
import tvm.relay as relay
import keras
import keras.datasets.mnist as mnist
import numpy as np
from tvm.contrib import graph_runtime
from tvm import rpc
from tvm.contrib import utils
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)

#Choose the quantize mode -> 1)"tvm_quant" 2)"prequant"
QUANTIZATION="tvm_quant"
#Load Lenet5 Model
def load_mod():
    MODEL_PATH  = "../model/lenet5/lenet5_model.h5"
    WEIGHT_PATH = "../model/lenet5/lenet5_weight.h5"
    keras_lenet5 = keras.models.load_model(MODEL_PATH)
    keras_lenet5.load_weights(WEIGHT_PATH)
    keras_lenet5.summary()
    return keras_lenet5

##########################################################################
#                Compile                 quantization                    #
#  Keras Model ---------> TVM Relay IR --------------> TVM QNN Relay IR  #
#                                                                        #
##########################################################################
shape_dict = {"conv2d_1_input": (1, 1, 28, 28)}
keras_lenet5 = load_mod()
mod, params = relay.frontend.from_keras(keras_lenet5, shape_dict)

#Save the original Relay Graph(before quant transformation)
with open("Lenet_Relay_origin.log",'w') as f:
    k_relay = mod.astext(show_meta_data = False)
    f.write(k_relay)

if QUANTIZATION == "tvm_quant":
    with relay.quantize.qconfig(    calibrate_mode="global_scale",
                                    global_scale=8.0,
                                    nbit_activation=16,
                                    dtype_activation="int16",
                                    skip_conv_layers=[],
                                    skip_dense_layer=False,
                                    partition_conversions="enabled"):
        mod = relay.quantize.quantize(mod, params)
        #viz = relay_viz.RelayVisualizer(mod)
        #viz.render()
        # Relay graph after quantization
        with open("Lenet_Relay_tvm_quant.log",'w') as f:
            k_relay = mod.astext(show_meta_data = False)
            f.write(k_relay)


#Compile the model
target = "c"
with relay.build_config(opt_level=0):
    graph, lib, param = relay.build(mod, target, params=params)

#with open("lib.log", 'w') as f:
#    f.write(lib.get_source())


#################################################################
#               quant                    Compile                #
#  Keras Model -------> Tensorflow Lite ---------> TVM RelayIR  #
#                                                               #
#################################################################
#Quantize the lenet5Keras model by tensoflow api
import tensorflow as tf
def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 28, 28, 1)
      yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_lenet5)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
lenet5_prequant = converter.convert()
with open("../model/lenet5/Lenet5_pre_quant" + ".tflite", 'wb') as f:
        f.write(lenet5_prequant)

#import tflite model
model_dir = "../model/lenet5"
tflite_model_file = os.path.join(model_dir, "Lenet5_pre_quant" + ".tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

#compile the model
shape_dict={"serving_default_conv2d_1_input:0": ( 1, 28, 28, 1)}
dtype_dict={"serving_default_conv2d_1_input:0": "int32"}
mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict, dtype_dict
)

#Save the compiled pre_quant model
with open("Lenet_pre_quant.log",'w') as f:
    k_relay = mod.astext(show_meta_data = False)
    f.write(k_relay)