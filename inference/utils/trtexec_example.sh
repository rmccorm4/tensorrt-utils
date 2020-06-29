#!/bin/bash

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

# Download sample pytorch->onnx script
wget https://gist.githubusercontent.com/rmccorm4/b72abac18aed6be4c1725db18eba4930/raw/3919c883b97a231877b454dae695fe074a1acdff/alexnet_onnx.py
# Install dependencies
python3 -m pip install torch==1.5.1 torchvision==0.6.1 onnx==1.6
# Export sample Alexnet model to ONNX with a dynamic batch dimension
python3 alexnet_onnx.py --opset=11

# Emulate "maxBatchSize" behavior from implicit batch engines by setting
# an optimization profile with min=(1, *shape), opt=max=(maxBatchSize, *shape)
MAX_BATCH_SIZE=32
INPUT_NAME="actual_input_1"

# Convert dynamic batch ONNX model to TRT Engine with optimization profile defined
#   --minShapes: kMIN shape
#   --optShapes: kOPT shape
#   --maxShapes: kMAX shape
#   --shapes:    # Inference shape - this is like context.set_binding_shape(0, shape)
trtexec --explicitBatch --onnx=alexnet_dynamic.onnx \
--minShapes=${INPUT_NAME}:1x3x224x224 \
--optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
--maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
--shapes=${INPUT_NAME}:1x3x224x224 \
--saveEngine=alexnet_dynamic.engine
