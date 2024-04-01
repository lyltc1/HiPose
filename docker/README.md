# Run Code in Docker

## Download Opencv
The normalSpeed function requires opencv3 as a necessary component.
1. Download opencv = 3.4.12 from [this link](https://github.com/opencv/opencv/archive/refs/tags/3.4.12.zip).
2. Download opencv_contrib = 3.4.12 from [this link](https://github.com/opencv/opencv_contrib/archive/refs/tags/3.4.12.zip).

## Download the image
We provide an image that can be downloaded using the command below.
```
docker pull lyltc1/hipose:latest
```
## Build images
Options: you can build the image by yourself.
```bash
cd HiPose/docker
bash build_docker.sh
```
## Usage
Pay attention to the dataset and output volume.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name HiPose \
-v path/to/dataset/:/home/dataset:ro \
-v path/to/output/:/home/HiPose/output:rw \
lyltc1/hipose:latest /bin/bash
```

## Update Code
update code to the latest version of HiPose repository
```
cd /home/HiPose
git pull
```

## prepare soft link
Assume that the dataset and Ground Truth (GT) from `zebrapose` have already been prepared.
```
ln -sf /home/dataset/pbr/lmo/* /home/HiPose/data/lmo/
ln -sf /home/dataset/zebrapose/data/lmo/* /home/HiPose/data/lmo/
```
Similarly, you can also prepare the datasets for ycbv and tless. Here's what the data/lmo/ directory should look like:
```
root@e8534fc7d360:/home/HiPose# ls data/lmo/
.gitkeep                            models_eval/                        camera.json                                             
dataset_info.md                     test/                               train_pbr_GT/
models/                             test_GT/                            train_pbr/
models_GT_color/                    test_targets_bop19.json             
```

## evaluate
```
cd /home/HiPose/hipose
`python test.py --cfg config/test_lmo_config.txt --obj_name ape --ckpt_file /path/to/lmo/lmo_convnext_ape/0_7824step86000 --eval_output /path/to/eval_output --new_solver_version True --region_bit 10`
```
The output `lmo_ape.csv` can be find in `output/exp/lmo_bop`

## train
```
cd /home/HiPose/hipose
python train.py --cfg config/train_lmo_config.txt --obj_name ape
```
