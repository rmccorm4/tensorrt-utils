#!/bin/bash

# Convenience script for building TensorRT OSS components inside
# of a TensorRT/TensorFlow container from [NGC](ngc.nvidia.com)
TAG=${1:-"master"}
ROOT=`pwd`

# Download OSS Components
git clone -b ${TAG} https://github.com/nvidia/TensorRT TensorRT
cd TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`

# Get most up to date ONNX parser if using OSS TensorRT master branch
if [[ ${TAG} -eq "master" ]]; then
  pushd parsers/onnx && git checkout master && git pull && popd
fi

# Install required libraries
apt-get update && apt-get install -y --no-install-recommends \
	libcurl4-openssl-dev \
	wget \
	zlib1g-dev \
	git \
	pkg-config \
	figlet

# Install CMake >= 3.13
pushd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh
chmod +x cmake-3.14.4-Linux-x86_64.sh
./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
rm ./cmake-3.14.4-Linux-x86_64.sh
popd

# Necessary in the nvcr.io/nvidia/tensorflow:19.10-py3 container due to some PATH/cmake issues
# Shouldn't be necessary in nvcr.io/nvidia/tensorrt:19.10-py3 container
export CMAKE_ROOT=/usr/share/cmake-3.14
source ~/.bashrc

# Set relevant env variables relative to NGC container paths
export TRT_RELEASE=/usr/src/tensorrt
export TRT_LIB_DIR=$TRT_RELEASE/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB_DIR

# Generate Makefiles and build
cd $TRT_SOURCE
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out
make -j$(nproc)

# Install OSS Components
make install && figlet "Success" || figlet "Failed"
cd ${ROOT}
