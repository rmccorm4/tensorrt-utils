#!/usr/bin/env python3

# Copyright 2020 NVIDIA Corporation
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

import argparse
from typing import Tuple, List

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def is_fixed(shape: Tuple[int]):
    return not is_dynamic(shape)


def is_dynamic(shape: Tuple[int]):
    return any(dim is None or dim < 0 for dim in shape)

  
def setup_binding_shapes(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    host_inputs: List[np.ndarray],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
):
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)

    assert context.all_binding_shapes_specified

    host_outputs = []
    device_outputs = []
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
        # Allocate buffers to hold output results after copying back to host
        buffer = np.empty(output_shape, dtype=np.float32)
        host_outputs.append(buffer)
        # Allocate output buffers on device
        device_outputs.append(cuda.mem_alloc(buffer.nbytes))

    return host_outputs, device_outputs


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    print("Engine/Binding Metadata")
    print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
    print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
    print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
    print("\tLast binding for profile {}: {}".format(profile_index, end_binding-1))

    # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)

    return input_binding_idxs, output_binding_idxs


def load_engine(filename: str):
    # Load serialized engine file into memory
    with open(filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def get_random_inputs(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    input_binding_idxs: List[int],
    seed: int = 42,
):
    # Input data for inference
    host_inputs = []
    print("Generating Random Inputs")
    print("\tUsing random seed: {}".format(seed))
    np.random.seed(seed)
    for binding_index in input_binding_idxs:
        # If input shape is fixed, we'll just use it
        input_shape = context.get_binding_shape(binding_index)
        input_name = engine.get_binding_name(binding_index)
        print("\tInput [{}] shape: {}".format(input_name, input_shape))
        # If input shape is dynamic, we'll arbitrarily select one of the
        # the min/opt/max shapes from our optimization profile
        if is_dynamic(input_shape):
            profile_index = context.active_optimization_profile
            profile_shapes = engine.get_profile_shape(profile_index, binding_index)
            print("\tProfile Shapes for [{}]: [kMIN {} | kOPT {} | kMAX {}]".format(input_name, *profile_shapes))
            # 0=min, 1=opt, 2=max, or choose any shape, (min <= shape <= max)
            input_shape = profile_shapes[1]
            print("\tInput [{}] shape was dynamic, setting inference shape to {}".format(input_name, input_shape))

        host_inputs.append(np.random.random(input_shape).astype(np.float32))

    return host_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, type=str,
                        help="Path to TensorRT engine file.")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Load a serialized engine into memory
    engine = load_engine(args.engine)
    print("Loaded engine: {}".format(args.engine))

    # Create context, this can be re-used
    context = engine.create_execution_context()
    # Profile 0 (first profile) is used by default
    context.active_optimization_profile = 0
    print("Active Optimization Profile: {}".format(context.active_optimization_profile))

    # These binding_idxs can change if either the context or the
    # active_optimization_profile are changed
    input_binding_idxs, output_binding_idxs = get_binding_idxs(
        engine, context.active_optimization_profile
    )
    input_names = [engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]
    
    # Generate random inputs based on profile shapes
    host_inputs = get_random_inputs(engine, context, input_binding_idxs, seed=args.seed)

    # Allocate device memory for inputs. This can be easily re-used if the
    # input shapes don't change
    device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
    # Copy host inputs to device, this needs to be done for each new input
    for h_input, d_input in zip(host_inputs, device_inputs):
        cuda.memcpy_htod(d_input, h_input)

    print("Input Metadata")
    print("\tNumber of Inputs: {}".format(len(input_binding_idxs)))
    print("\tInput Bindings for Profile {}: {}".format(context.active_optimization_profile, input_binding_idxs))
    print("\tInput names: {}".format(input_names))
    print("\tInput shapes: {}".format([inp.shape for inp in host_inputs]))

    # This needs to be called everytime your input shapes change
    # If your inputs are always the same shape (same batch size, etc.),
    # then you will only need to call this once
    host_outputs, device_outputs = setup_binding_shapes(
        engine, context, host_inputs, input_binding_idxs, output_binding_idxs,
    )
    output_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]

    print("Output Metadata")
    print("\tNumber of Outputs: {}".format(len(output_binding_idxs)))
    print("\tOutput names: {}".format(output_names))
    print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
    print("\tOutput Bindings for Profile {}: {}".format(context.active_optimization_profile, output_binding_idxs))
    
    # Bindings are a list of device pointers for inputs and outputs
    bindings = device_inputs + device_outputs

    # Inference
    context.execute_v2(bindings)

    # Copy outputs back to host to view results
    for h_output, d_output in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh(h_output, d_output)

    # View outputs
    print("Inference Outputs:", host_outputs)

    # Cleanup (Can also use context managers instead)
    del context
    del engine

if __name__ == "__main__":
    main()
