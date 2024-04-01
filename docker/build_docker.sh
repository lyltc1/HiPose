#!/bin/bash

if [ ! -f "./opencv-3.4.12.zip" ]; then
    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip -O opencv-3.4.12.zip
fi

if [ ! -f "./opencv_contrib-3.4.12.zip" ]; then
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip -O opencv_contrib-3.4.12.zip
fi

docker build -t lyltc1/hipose:latest .

