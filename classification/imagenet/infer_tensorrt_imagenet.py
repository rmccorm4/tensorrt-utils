#!/usr/bin/env python3

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import argparse
import PIL.Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # To automatically manage CUDA context creation and cleanup


def load_normalized_test_case(test_images, pagelocked_buffer, preprocess_func):
    # Expected input dimensions
    C, H, W = (3, 224, 224)
    # Normalize the images, concatenate them and copy to pagelocked memory.
    data = np.asarray([preprocess_func(PIL.Image.open(img), C, H, W) for img in test_images]).flatten()
    np.copyto(pagelocked_buffer, data)


class HostDeviceMem(object):
    r""" Simple helper data class that's a little nicer to use than a 2-tuple.
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int):
    print('Allocating buffers ...')

    inputs = []
    outputs = []
    dbindings = []

    stream = cuda.Stream()

    for binding in engine:
        size = batch_size * trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        dbindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings, stream


def infer(engine_path, preprocess_func, batch_size=8, input_images=[], labels=[], num_classes=3):
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings, stream = allocate_buffers(engine, batch_size)

        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            test_images = np.random.choice(input_images, size=batch_size)
            load_normalized_test_case(test_images, inputs[0].host, preprocess_func)

            inp = inputs[0]
            # Transfer input data to the GPU.
            cuda.memcpy_htod(inp.device, inp.host)

            # Run inference.
            context.execute(batch_size, dbindings)

            out = outputs[0]
            # Transfer predictions back to host from GPU
            cuda.memcpy_dtoh(out.host, out.device)
            out_np = np.array(out.host)

            # Split 1-D output of length N*labels into 2-D array of (N, labels)
            batch_outs = np.array(np.split(out_np, batch_size))
            for test_image, batch_out in zip(test_images, batch_outs):
                topk_indices = np.argsort(batch_out)[-1*num_classes:][::-1]
                preds = labels[topk_indices]
                probs = batch_out[topk_indices]
                print("Input image:", test_image)
                for pred, prob in zip(preds, probs):
                    print("\tPrediction: {:30} Probability: {:0.2f}".format(pred, prob))



def get_inputs(filename=None, directory=None, allowed_extensions=(".jpeg", ".jpg", ".png")):
    filenames = []
    if filename:
        filenames.append(filename)
    if directory:
        dir_files = [path for path in glob.iglob(os.path.join(directory, "**"), recursive=True) if os.path.isfile(path) and path.lower().endswith(allowed_extensions)]
        filenames.extend(dir_files)

    if len(filenames) <= 0:
        raise ValueError("ERROR: No valid inputs given.")

    return filenames 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on TensorRT engines for Imagenet-based Classification models.')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine {resnet50, vgg16, inception_v1, mobilenetv2-1.0...}')
    parser.add_argument('-f', '--file', default=None, type=str, help="Path to input image.")
    parser.add_argument('-d', '--directory', default=None, type=str, help="Path to directory of input images.")
    parser.add_argument('-n', '--num_classes', default=3, type=int, help="Top-K predictions to output.")
    parser.add_argument('-b', '--batch_size', default=8, type=int, help="Number of inputs to send in parallel (up to max batch size of engine).")
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="Name of function defined in 'processing.py' to use for pre-processing calibration data.")
    args = parser.parse_args()

    input_images = get_inputs(args.file, args.directory)
    with open("imagenet1k_labels.txt", "r") as f:
        labels = np.array(f.read().splitlines())

    # Choose pre-processing function for inference inputs
    import processing
    if args.preprocess_func is not None:
        preprocess_func = getattr(processing, args.preprocess_func)
    else:
        preprocess_func = processing.preprocess_imagenet

    infer(args.engine, preprocess_func, batch_size=args.batch_size, input_images=input_images, labels=labels, num_classes=args.num_classes)
