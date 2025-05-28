ARG PYTORCH="2.4.1"
ARG CUDA="12.1"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

# Basic setting 
RUN apt-get update

# Install normalSpeed
RUN apt-get install -y build-essential unzip
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
WORKDIR /home
COPY opencv-3.4.12.zip /home/opencv-3.4.12.zip
COPY opencv_contrib-3.4.12.zip /home/opencv_contrib-3.4.12.zip
RUN unzip opencv-3.4.12.zip
RUN unzip opencv_contrib-3.4.12.zip
WORKDIR /home/opencv-3.4.12
RUN mkdir build
WORKDIR /home/opencv-3.4.12/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib-3.4.12/modules/ ..
RUN make -j7 && make install
WORKDIR /home
RUN git clone https://github.com/hfutcgncas/normalSpeed.git
WORKDIR /home/normalSpeed/normalSpeed
RUN pip3 install "pybind11[global]"
RUN python3 setup.py install --user

# Install other dependency
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y python3-tk
RUN apt-get install -y libgl1
RUN pip install --no-cache-dir \
    tqdm==4.66.5 \
    tensorboard==2.19.0 \
    opencv-python-headless==4.11.0.86 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    open3d-cpu==0.19.0 \
    albumentations==2.0.7 \
    timm==1.0.15 

# clone the code
WORKDIR /home
RUN git clone https://github.com/lyltc1/HiPose.git

# Install bop_toolkit in editing mode
WORKDIR /home/HiPose/hipose/bop_toolkit
RUN pip install -r requirements.txt -e .

# Install Randla-net
WORKDIR /home/HiPose/hipose/models/RandLA

RUN sh compile_op.sh
# Clean everything
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
WORKDIR /home
RUN rm opencv-3.4.12.zip
RUN rm opencv_contrib-3.4.12.zip
RUN rm -r opencv-3.4.12
RUN rm -r opencv_contrib-3.4.12

WORKDIR /home/HiPose


