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
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version to generate models with.")
args = parser.parse_args()

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = [ "actual_input_1" ] #+ [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Fixed Shape
torch.onnx.export(model, dummy_input, "alexnet_fixed.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names)

# Dynamic Shape
dynamic_axes = {"actual_input_1":{0:"batch_size"}, "output1":{0:"batch_size"}}
print(dynamic_axes)
torch.onnx.export(model, dummy_input, "alexnet_dynamic.onnx", verbose=True, opset_version=args.opset,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes)
