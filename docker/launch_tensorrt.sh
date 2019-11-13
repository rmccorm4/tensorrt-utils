#!/bin/bash

# Convenience script to launch an interactive NGC TensorRT 
# container with mounted code and data

DATASET="/imagenet"

nvidia-docker run -it \
    -v ${PWD}/../:/mnt \
    -v ${DATASET}:${DATASET} \
    --workdir /mnt \
    nvcr.io/nvidia/tensorrt:19.10-py3 \
    bash 
