# ImageNet Models: ONNX -> TensorRT

This directory contains some helper scripts for creating TensorRT engines from
various ONNX classification models based on Imagenet data using the Python API. 

These scripts were last tested using the 
[NGC TensorRT Container Version 19.10-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).

## ONNX Model Zoo

To name a few models taken from the [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/classification)
that should work pretty well with these scripts:
* [ResNet50](https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz)
* [MobileNetV2](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz)
* [InceptionV1](https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz)
* [VGG16](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz)

> **NOTE**: For creating FP32/FP16 engines, **it will likely be much simpler to just use `trtexec --onnx=model.onnx ...`**.
> These `python` scripts are mostly just convenient for doing `INT8` calibration, because `trtexec` currently just creates
> `INT8` engines for the sake of benchmarking, and doesn't preserve the accuracy of the model. The FP32/FP16 options just
> come along for free.

## INT8 Calibration

See [ImagenetCalibrator.py](ImagenetCalibrator.py) for a reference implementation
of TensorRT's [IInt8EntropyCalibrator2](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html).

This class can be tweaked to work for other kinds of models, inputs, etc.

### Pre-processing

In order to calibrate your model correctly, you should `pre-process` your data the same way
that you would during inference. You can pass in a `preprocess_func` to the constructor
of `ImagenetCalibrator(..., preprocess_func=<function_name>, ...)`,  where `<function_name>`
is a string, corresponding to the name of a pre-processing function defined inside of
`processing.py`. You can add your own pre-processing functions to `processing.py` and pass
the function name into the constructor accordingly.
