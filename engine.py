import torch
# import torchaudio
import soundfile as sf
import numpy as np
import kaldi_native_fbank as knf
from torch.nn.utils.rnn import pad_sequence
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import common
import torch.nn.functional as F
from collections import defaultdict
import math
from typing import List, Tuple
import time


TRT_LOGGER = trt.Logger()
# onnx_file_path = '/disc1/yzp/code/wenet/final_go/whole_model_20.onnx'
# engine_file_path = "/disc1/yzp/code/wenet/final_go/whole_model_20.trt"


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp



# def get_forward(chunk_xs,offset,att_cache,cnn_cache,stream,cur):
#     engine = get_engine(onnx_file_path,engine_file_path)
#     context = engine.create_execution_context()
#     # context.set_optimization_profile_async(0, stream.handle)
#     # print(context.all_binding_shapes_specified)

#     context.set_binding_shape(0,chunk_xs.shape)
#     context.set_binding_shape(1,offset.shape)
#     context.set_binding_shape(2,att_cache.shape)
#     context.set_binding_shape(3,cnn_cache.shape)
#     # print(context.all_binding_shapes_specified)
#     inputs, outputs, bindings, stream = allocate_buffers(engine,context,stream)
#     # print('x')
#     inputs[0].host=chunk_xs
#     inputs[1].host=offset
#     inputs[2].host=att_cache
#     inputs[3].host=cnn_cache
#     trt_outputs=common.do_inference_v2(context,bindings=bindings,inputs=inputs, outputs=outputs, stream=stream)
#     out_shapes=[(12,4,12*((int(cur/48))+2),128),(12,1,256,7),(12,4233)]
#     trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
#     return trt_outputs


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine,context):
    inputs = []
    outputs = []
    bindings = []
    for binding in range(len(engine)):
        size = trt.volume(context.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings 


def get_engine(onnx_file_path, 
                num_layer,
                num_head,
                decoding_window,
                left_chunk,
                input_fbank_dim,
                decoding_chunk_size,
                att_dim,
                cnn_dim,
                att_out_dim,
                cnn_out_dim,
                engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]
            #设置输入维度，设置为动态维度模式，输入维度数必须大于2
            input_chunk = network.get_input(0)
            input_offet = network.get_input(1)
            input_att_cache=network.get_input(2)
            input_cnn_cache=network.get_input(3)


            # network.add_input("input", trt.float32,(1, -1, -1))
            # network.get_input(0).shape=[1,-1,80]
            # network.get_input(1).shape=[1,]
            # network.get_input(2).shape=[12,4,-1,128]
            # network.get_input(3).shape=[12,1,256,7]
            


            profile = builder.create_optimization_profile()
            profile.set_shape(input_chunk.name, (1,decoding_window,input_fbank_dim), ( 1,decoding_window,input_fbank_dim), (1,decoding_window,input_fbank_dim)) 
            profile.set_shape(input_offet.name, (1,1),(1,1), (1,1))

            profile.set_shape(input_att_cache.name, (num_layer,num_head,12,att_out_dim), (num_layer,num_head,att_dim,att_out_dim), (num_layer,num_head,240,att_out_dim)) 
            profile.set_shape(input_cnn_cache.name, (num_layer,1,cnn_out_dim,cnn_dim), (num_layer,1,cnn_out_dim,cnn_dim), (num_layer,1,cnn_out_dim,cnn_dim)) 



            config.add_optimization_profile(profile)
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)




            # network.get_input(0).shape=[1,12]
            # engine = builder.build_cuda_engine(network)


            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        # print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()




class SP_engine(object):
    def __init__(self, 
                onnx_file_path,
                num_layer,
                num_head,
                decoding_window,
                left_chunk,
                input_fbank_dim,
                decoding_chunk_size,
                att_dim,
                cnn_dim,
                att_out_dim,
                cnn_out_dim,
                final_out_dim,
                engine_file_path="",):

        self.engine=get_engine(onnx_file_path, 
                            num_layer,
                            num_head,
                            decoding_window,
                            left_chunk,
                            input_fbank_dim,
                            decoding_chunk_size,
                            att_dim,
                            cnn_dim,
                            att_out_dim,
                            cnn_out_dim,
                            engine_file_path)
        self.context = self.engine.create_execution_context()
        self.stream= cuda.Stream()
        self.decoding_chunk_size=decoding_chunk_size
        self.final_out_dim=final_out_dim
        self.att_dim=att_dim
        self.cnn_dim=cnn_dim
        self.att_out_dim=att_out_dim
        self.cnn_out_dim=cnn_out_dim
        self.num_layer=num_layer
        self.num_head=num_head
        self.left_chunk=left_chunk

        # self.inputs,self.outputs,self.bindings=allocate_buffers(self.engine,self.context)
    def infer(self,
            chunk_xs,
            offset,
            att_cache,
            cnn_cache,
            cur,


            ):
        context=self.context
        context.set_binding_shape(0,chunk_xs.shape)
        context.set_binding_shape(1,offset.shape)
        context.set_binding_shape(2,att_cache.shape)
        context.set_binding_shape(3,cnn_cache.shape)
        la=att_cache.shape[2]
        if la==self.decoding_chunk_size*self.left_chunk:
            la=la+self.decoding_chunk_size
        else:
            la=la+self.decoding_chunk_size
        # print(context.all_binding_shapes_specified)
        inputs,outputs,bindings=allocate_buffers(self.engine,self.context)
        # print('x')
        inputs[0].host=chunk_xs
        inputs[1].host=offset
        inputs[2].host=att_cache
        inputs[3].host=cnn_cache
        #output[0]为att_cache， output[1]为cnn_cache， output[2]为probs
        trt_outputs=common.do_inference_v2(context,bindings=bindings,inputs=inputs, outputs=outputs, stream=self.stream)
        out_shapes=[(self.num_layer,self.num_head,la,self.att_out_dim),(self.num_layer,1,self.cnn_out_dim,self.cnn_dim),(self.decoding_chunk_size,self.final_out_dim)]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        return trt_outputs