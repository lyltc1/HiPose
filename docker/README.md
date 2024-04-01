# README

## Download
1. Download opencv = 3.4.12 from https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip
2. Download opencv_contrib = 3.4.12 from https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip


## Build 
```bash
cd HiPose/docker
bash build_docker.sh
```
## Usage
Pay attention to the dataset and output volume.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all --shm-size 12G --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --name HiPose -v /home/lyl/dataset/:/home/dataset:ro -v /home/lyl/git/output/:/home/HiPose/output:rw lyltc1/hipose:latest /bin/bash
```

## prepare soft link
Assume the dataset and GT from zebrapose has been prepared.
```
ln -sf /home/dataset/pbr/lmo/* /home/HiPose/data/lmo/
ln -sf /home/dataset/zebrapose/data/lmo/* /home/HiPose/data/lmo/
```

## install bop_toolkit
Note that the version inside requirements.txt is modified inorder that the installation can be success
```
pip install Cython==0.29.24
cd /home/HiPose/hipose/bop_toolkit
pip install -r requirements.txt -e .  
```

## build RandLA
```
cd /home/z3d/zebrapose/models/RandLA
sh compile_op.sh
cp /home/z3d/zebrapose/models/RandLA/utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.10-linux-x86_64.egg/nearest_neighbors.cpython-310-x86_64-linux-gnu.so /home/z3d/zebrapose/models/RandLA/utils/nearest_neighbors/lib/python/nearest_neighbors.cpython-310-x86_64-linux-gnu.so
cp /home/z3d/zebrapose/models/RandLA/utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.10-linux-x86_64.egg/nearest_neighbors.py /home/z3d/zebrapose/models/RandLA/utils/nearest_neighbors/lib/python/nearest_neighbors.py
```

## evaluate
```
cd /home/z3d/zebrapose
python test.py --cfg config/config_zebra3d/lmo_zebra3D_32_no_hier_lmo_bop_gdrnpp_.txt --obj_name ape --ckpt_file /home/dataset/z3d/lmo_zebra3D_32_no_hier_lmo_bop_ape/0_7668step37000 --ignore_bit 0 --eval_output_path /home/z3d/output/
```

## Docker Usage
```
docker stop \z3d
docker rm \z3d
```
