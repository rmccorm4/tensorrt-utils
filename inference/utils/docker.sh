#!/bin/bash
nvidia-docker run -it -v `pwd`:/mnt -w /mnt nvcr.io/nvidia/tensorrt:20.06-py3
