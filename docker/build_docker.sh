#!/bin/bash

# manually download opencv = 3.4.12 from https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip
if [ ! -f "./opencv-3.4.12.zip" ]; then
    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip -O opencv-3.4.12.zip
fi
# manually download opencv_contrib = 3.4.12 from https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip
if [ ! -f "./opencv_contrib-3.4.12.zip" ]; then
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip -O opencv_contrib-3.4.12.zip
fi

PYTORCH_VERSION="2.4.1"
CUDA_VERSION="12.1"
CUDNN_VERSION="9"

docker build \
    --build-arg PYTORCH=${PYTORCH_VERSION} \
    --build-arg CUDA=${CUDA_VERSION} \
    --build-arg CUDNN=${CUDNN_VERSION} \
    -t lyltc1/hipose:pytorch${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION} .
