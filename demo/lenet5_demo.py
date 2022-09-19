# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Keras Models
=====================
**Author**: `Yuwei Hu <https://Huyuwei.github.io/>`_

This article is an introductory tutorial to deploy keras models with Relay.

For us to begin with, keras should be installed.
Tensorflow is also required since it's used as the default backend of keras.

A quick solution is to install via pip

.. code-block:: bash

    pip install -U keras --user
    pip install -U tensorflow --user

or please refer to official site
https://keras.io/#installation
"""
import tvm
from tvm import te
import tvm.relay as relay
import keras
import keras.datasets.mnist as mnist
import numpy as np
from tvm.contrib import graph_runtime
from tvm import rpc
from tvm.contrib import utils

QUANTIZATION = "ON"
RPC = "OFF"
POP_LOOP = 0 # 0~6
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
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
data = x_test[2]
#data = gray.reshape(28,28,1)
#data = data/255
#np.testing.assert_allclose(data_test,data,atol=1e-1)

data = data.reshape(1, 28, 28, 1)
data = np.transpose(data, (0, 3, 1, 2))

# Load pretrained model
def load_mod():
    MODEL_PATH  = "../model/lenet5/lenet5_model.h5"
    WEIGHT_PATH = "../model/lenet5/lenet5_weight.h5"
    keras_lenet5 = keras.models.load_model(MODEL_PATH)
    keras_lenet5.load_weights(WEIGHT_PATH)
    for x in range(POP_LOOP):
        keras_lenet5.pop()
    keras_lenet5.summary()
    return keras_lenet5

######################################################################
# Compile the model with Relay
# ----------------------------
# convert the keras model(NHWC layout) to Relay format(NCHW layout).
shape_dict = {"conv2d_1_input": data.shape}
keras_lenet5 = load_mod()
mod, params = relay.frontend.from_keras(keras_lenet5, shape_dict)
# Original Relay graph before transformation
with open("Lenet_Relay_origin.log",'w') as f:
    k_relay = mod.astext(show_meta_data = False)
    f.write(k_relay)

# Quantization of the model
if QUANTIZATION == "ON":
    with relay.quantize.qconfig(    calibrate_mode="global_scale",
                                    global_scale=8.0,
                                    nbit_activation=16,
                                    dtype_activation="int16",
                                    skip_conv_layers=[],
                                    skip_dense_layer=False):
        mod = relay.quantize.quantize(mod, params)
        # Relay graph after quantization
        with open("Lenet_Relay_quant.log",'w') as f:
            k_relay = mod.astext(show_meta_data = False)
            f.write(k_relay)

# Relay graph annotation
ANNOTATION = "ON"
ANNOTATION_TARGET = "CASDLA"
# ANNOTATION_TARGET = "CLib"
if ANNOTATION == "ON" or RPC == "ON":
    from tvm.relay import transform

    if (ANNOTATION_TARGET=="CASDLA"):
        from tvm.relay.op.contrib.CASDLA import CASDLA_pattern_table as pattern_table
    elif (ANNOTATION_TARGET=="CLib"):
        from tvm.relay.op.contrib.CLib import CLib_pattern_table as pattern_table
    else:
        print("Unknown annotation target")
        quit()
    
    amod = transform.MergeComposite(pattern_table())(mod)
    amod = transform.AnnotateTarget([ANNOTATION_TARGET])(amod)
    with open("Lenet_Relay_annotate.log",'w') as f:
        opt_relay = amod.astext(show_meta_data = False)
        f.write(opt_relay)
    amod = transform.MergeCompilerRegions()(amod)
    with open("Lenet_Relay_merge.log",'w') as f:
        opt_relay = amod.astext(show_meta_data = False)
        f.write(opt_relay)
    amod = transform.PartitionGraph()(amod)

    # Relay graph after annotation & optimization
    with open("Lenet_Relay_opt.log",'w') as f:
        opt_relay = amod.astext(show_meta_data = False)
        f.write(opt_relay)

########### compile the acc model #############
target = "c"
ctx = tvm.cpu()
if ANNOTATION == "ON":
    with tvm.transform.PassContext(opt_level=0,config={'tir.disable_vectorize':True,'relay.FuseOps.max_depth':0},disabled_pass=["AlterOpLayout"]):
        agraph, alib, aparam = relay.build(amod, target=target, params=params)

    # compile the library
    alib.export_library("./alib.so")
    alib = tvm.runtime.load_module("./alib.so")

    acc_m = graph_runtime.create(agraph, alib, ctx)
    acc_m.set_input("conv2d_1_input", tvm.nd.array(data.astype("float32")))
    acc_m.set_input(**aparam)
    acc_m.run()
    acc_out = acc_m.get_output(0)
    acc_out = acc_out.asnumpy()
    print(acc_out)
    # if np.testing.assert_allclose(llvm_out,acc_out,atol=1e-1)==None:
    #     print("Pass!!!")
else:
    with relay.build_config(opt_level=0):
        graph, lib, param = relay.build(mod, target=target,params=params)
    with tvm.transform.PassContext(opt_level=0,config={'tir.disable_vectorize':True,'relay.FuseOps.max_depth':0},disabled_pass=["AlterOpLayout"]):
        graph, lib, param = relay.build(mod, target=target, params=params)

    # compile the library
    #print(lib.imported_modules[0].get_source())
    lib.export_library("./lib.so")
    lib = tvm.runtime.load_module("./lib.so")

    llvm_m = graph_runtime.create(graph, lib, ctx)
    llvm_m.set_input("conv2d_1_input", tvm.nd.array(data.astype("float32")))
    llvm_m.set_input(**param)
    llvm_m.run()
    llvm_out = llvm_m.get_output(0)
    llvm_out = llvm_out.asnumpy()
    print(llvm_out)


# Terminate CASDLA if annotation is on and target is CASDLA
if (ANNOTATION=="ON" and ANNOTATION_TARGET=="CASDLA"):
    # Try acquire shared memory on python\
    ## Import ctypes to use shared memory on python
    from ctypes import *

    ## CTypes data structure to sync with the shared memory on accelerator
    class CAS_SYSBUS_DATA(Structure):
        _fields_ = [("addr", c_uint), 
                    ("data", c_ulonglong),
                    ("ack", c_uint),
                    ("cmd", c_uint),
                    ("isr", c_uint),
                    ("syncdata", c_ubyte*1024),
                    ("synclength", c_uint32)]

    # Hard-wired shared memory ID for CAS_SYSBRG
    CAS_SYSBRG_SMID = 0x8000
    sizeOfSysBusData = sizeof(CAS_SYSBUS_DATA)
    # Load library for shmget/shmat
    try:  
        rt = CDLL('librt.so', use_errno=True)  
    except:  
        rt = CDLL('librt.so.1', use_errno=True)
    # Get the funcitons `shmget`, `shmat`
    shmget = rt.shmget  
    shmget.argtypes = [c_int, c_size_t, c_int]  
    shmget.restype = c_int  
    shmat = rt.shmat  
    shmat.argtypes = [c_int, POINTER(c_void_p), c_int]  
    shmat.restype = c_void_p  

    # Get shared memory with ID, size and flags
    shmid = shmget(c_int(CAS_SYSBRG_SMID), c_size_t(sizeOfSysBusData), c_int(0o1666)) 

    # Termination signal for hardware -> read the last address on bus
    pSysBusDataShm = None
    CAS_SYSBUS_CMD_EXIT	= 0x04
    sysBusDataToWrite = CAS_SYSBUS_DATA(c_uint(0), c_ulonglong(0), c_uint(0), c_uint(CAS_SYSBUS_CMD_EXIT))
    if shmid < 0:
        print("Failed to open shared memory")
        print(get_errno())
    # On successfully opened shared memory, move the signal to the bridge
    else:
        pSysBusDataShm = shmat(shmid, None, 0)
        memmove(pSysBusDataShm, pointer(sysBusDataToWrite), sizeOfSysBusData)
