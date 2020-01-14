#!/bin/bash
# Move sampleRepro into TensorRT samples dir
cp -r sampleRepro /usr/src/tensorrt/samples
# Add sampleRepro to Makefile and comment out other samples
sed -i 's/samples=/samples=sampleRepro #/' /usr/src/tensorrt/samples/Makefile
# Build sampleRepro
pushd /usr/src/tensorrt/samples && make && popd
# Run sampleRepro
sample_repro
