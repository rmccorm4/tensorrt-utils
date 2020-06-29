# TensorRT Inference Example (Python API)

A generic inference script example to handle TensorRT engines created from both fixed-shape
and dynamic-shape ONNX models. 

This script will detect the input/output names/shapes and
generate random inputs to do inference on. It will also dump a lot of relevant metadata
and information that happens along the way to clarify the steps that happen when doing
inference with TensorRT.

## Setup

1. (Optional) Start a TensorRT 7.1 Docker container using the NGC 20.06 TensorRT Release:
```
# Mount current directory to /mnt and use /mnt as workspace inside container
nvidia-docker run -it -v `pwd`:/mnt -w /mnt nvcr.io/nvidia/tensorrt:20.06-py3
```

2. (Optional) Download and run a sample script to:
  * Generate a fixed-shape Alexnet ONNX model
  * Generate a dynamic-shape Alexnet ONNX model

```
# Download sample pytorch->onnx script
wget https://gist.githubusercontent.com/rmccorm4/b72abac18aed6be4c1725db18eba4930/raw/3919c883b97a231877b454dae695fe074a1acdff/alexnet_onnx.py
# Install dependencies
python3 -m pip install torch==1.5.1 torchvision==0.6.1 onnx==1.6
# Export sample Alexnet model to ONNX with a dynamic batch dimension
python3 alexnet_onnx.py --opset=11
```

3. (Optional) Use `trtexec` to:
  * Create a TensorRT engine from the **fixed-shape** Alexnet ONNX model in previous step

```
# Create TensorRT engine from fixed-shape ONNX model
trtexec --explicitBatch \
        --onnx=alexnet_fixed.onnx \
        --saveEngine=alexnet_fixed.engine
```

4. (Optional) Use `trtexec` to:
  * Create a TensorRT engine from the **dynamic-shape** Alexnet ONNX model in previous step

```
# Emulate "maxBatchSize" behavior from implicit batch engines by setting
# an optimization profile with min=(1, *shape), opt=max=(maxBatchSize, *shape)
MAX_BATCH_SIZE=32
INPUT_NAME="actual_input_1"

# Convert dynamic batch ONNX model to TRT Engine with optimization profile defined
#   --minShapes: kMIN shape
#   --optShapes: kOPT shape
#   --maxShapes: kMAX shape
#   --shapes:    # Inference shape - this is like context.set_binding_shape(0, shape)
trtexec --onnx=alexnet_dynamic.onnx \
        --explicitBatch \
        --minShapes=${INPUT_NAME}:1x3x224x224 \
        --optShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --maxShapes=${INPUT_NAME}:${MAX_BATCH_SIZE}x3x224x224 \
        --shapes=${INPUT_NAME}:1x3x224x224 \
        --saveEngine=alexnet_dynamic.engine
```

## Inference

1. Run a generic python script that can handle both fixed-shape and dynamic-shape
   TensorRT engines. 
  * This script generates random input data
  * For dynamic shape engines, this script defaults to input shapes of the engine's 
    first optimization profile's (`context.active_optimization_profile=0`) kOPT shape

> **NOTE**: This script is not meant for peak performance, but is meant to serve as a detailed
  example to demonstrate all of the moving parts involved in inference, especially for
  dynamic shape engines.

### Fixed-shape Engine Example

```
root@49b19ca81d38:/mnt# python3 infer.py -e alexnet_fixed.engine   
Loaded engine: alexnet_fixed.engine
Active Optimization Profile: 0
Engine/Binding Metadata
        Number of optimization profiles: 1
        Number of bindings per profile: 2
        First binding for profile: 0
        Last binding for profile: 0
Generating Random Inputs
        Input [actual_input_1] shape: (10, 3, 224, 224)
Input Metadata
        Number of Inputs: 1
        Input Bindings for Profile 0: [0]
        Input names: ['actual_input_1']
        Input shapes: [(10, 3, 224, 224)]
Output Metadata
        Number of Outputs: 1
        Output names: ['output1']
        Output shapes: [(10, 1000)]
        Output Bindings for Profile 0: [1]
Inference Outputs: [array([[-1.9423429 , -0.31228915, -0.11592922, ..., -1.8848201 ,
        -1.967107  ,  1.77118   ],
       [-2.067131  , -0.1413779 , -0.10807458, ..., -1.8306588 ,
        -2.0896611 ,  1.8038476 ],
       [-1.9210347 , -0.59420735, -0.46031308, ..., -1.8159304 ,
        -1.953881  ,  1.7182006 ],
       ...,
       [-2.091892  , -0.24702555,  0.01917603, ..., -2.0384629 ,
        -2.075258  ,  1.877627  ],
       [-2.1854897 , -0.48040774, -0.19558612, ..., -1.9172117 ,
        -2.1862085 ,  1.877256  ],
       [-2.2054253 , -0.29715982, -0.15480372, ..., -1.5839869 ,
        -1.941939  ,  1.9311339 ]], dtype=float32)]
```

### Dynamic-shape Engine Example

```
root@49b19ca81d38:/mnt# python3 infer.py -e alexnet_dynamic.engine 
Loaded engine: alexnet_dynamic.engine
Active Optimization Profile: 0
Engine/Binding Metadata
        Number of optimization profiles: 1
        Number of bindings per profile: 2
        First binding for profile: 0
        Last binding for profile: 0
Generating Random Inputs
        Input [actual_input_1] shape: (-1, 3, 224, 224)
        Profile Shapes for [actual_input_1]: [kMIN (1, 3, 224, 224) | kOPT (32, 3, 224, 224) | kMAX (32, 3, 224, 224)]
        Input [actual_input_1] shape was dynamic, setting inference shape to (32, 3, 224, 224)
Input Metadata
        Number of Inputs: 1
        Input Bindings for Profile 0: [0]
        Input names: ['actual_input_1']
        Input shapes: [(32, 3, 224, 224)]
Output Metadata
        Number of Outputs: 1
        Output names: ['output1']
        Output shapes: [(32, 1000)]
        Output Bindings for Profile 0: [1]
Inference Outputs: [array([[-2.1023765 , -0.3775048 , -0.17939475, ..., -1.8248225 ,
        -2.042846  ,  2.0328057 ],
       [-1.737236  , -0.54789805, -0.33429265, ..., -1.7502917 ,
        -2.173114  ,  1.699208  ],
       [-1.9979393 , -0.467514  , -0.3265229 , ..., -1.8016856 ,
        -2.025305  ,  1.6176162 ],
       ...,
       [-1.9945843 , -0.42376897,  0.11915839, ..., -1.7103177 ,
        -2.0614984 ,  1.7928884 ],
       [-2.180842  , -0.52559745, -0.12792507, ..., -1.9101417 ,
        -2.132482  ,  1.8081576 ],
       [-2.011236  , -0.29210696, -0.2629762 , ..., -1.8188174 ,
        -2.0216613 ,  1.884306  ]], dtype=float32)]
```
