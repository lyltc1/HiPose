ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

# Install git
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >>/etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >>/etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >>/etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y git

# clone the code
WORKDIR /home
RUN git clone https://github.com/lyltc1/HiPose.git

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
RUN apt-get install -y libgl1
RUN pip install tqdm
RUN pip install opencv-python-headless
RUN pip install imgaug
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install open3d-cpu
RUN pip install timm
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y python3-tk

# Install bop_toolkit in editing mode
WORKDIR /home/HiPose/hipose/bop_toolkit
RUN pip install -r requirements.txt -e .

# Install Randla-net
WORKDIR /home/HiPose/hipose/models/RandLA
RUN sh compile_op.sh
RUN cp /home/HiPose/hipose/models/RandLA/utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.10-linux-x86_64.egg/nearest_neighbors.cpython-310-x86_64-linux-gnu.so /home/HiPose/hipose/models/RandLA/utils/nearest_neighbors/lib/python/nearest_neighbors.cpython-310-x86_64-linux-gnu.so
RUN cp /home/HiPose/hipose/models/RandLA/utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.10-linux-x86_64.egg/nearest_neighbors.py /home/HiPose/hipose/models/RandLA/utils/nearest_neighbors/lib/python/nearest_neighbors.py

# Clean everything
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
WORKDIR /home
RUN rm opencv-3.4.12.zip
RUN rm opencv_contrib-3.4.12.zip
RUN rm -r opencv-3.4.12
RUN rm -r opencv_contrib-3.4.12




