# Sample C++ Repro

This is a mini TensorRT C++ API example that parses a simple ONNX model, 
creates a TensorRT engine from the ONNX model, and serializes it to a file for future use.

This sample mostly exists just for a quick way to create, build, and run a C++ repro inside a TensorRT
container. Simply follow these steps:

1. Start up a TensorRT container, such as one from [NGC](ngc.nvidia.com):
```bash
nvidia-docker run -it -v ${PWD}:/mnt --workdir=/mnt nvcr.io/nvidia/tensorrt:19.12-py3
```

2. (Optional) Add/edit your sample code in `sampleRepro/sampleRepro.cpp`:
```bash
vim sampleRepro/sampleRepro.cpp
```

The default sampleRepro.cpp is basically the C++ API equivalent to this:
```bash
trtexec --onnx=model.onnx --explicitBatch --fp16 --saveEngine=model.engine
```

3. Build and run the sample:
```bash
./run_sample.sh
```

4. Test the engine:
```bash
trtexec --loadEngine=model.engine
```
