# HiPose

The implementation of the paper 'HiPose: Hierarchical Binary Surface Encoding and Correspondence Pruning for RGB-D 6DoF Object Pose Estimation' (CVPR2024). [`ArXiv`](https://arxiv.org/abs/2311.12588)

![pipeline](pic/overview.png)

## Environment
- CUDA 11.1 or 11.6
- torch 1.13.1 and torchvision 0.14.1
- Open3d
- [normalSpeed](https://github.com/hfutcgncas/normalSpeed), a fast and light-weight normal map estimator
- [RandLA-Net](https://github.com/qiqihaer/RandLA-Net-pytorch) operators
- [bop_toolkit](https://github.com/thodan/bop_toolkit)

Setting up the environment can be tedious, so we've provided a Dockerfile to simplify the process. Please refer to the [README](./docker/README.md) in the Docker directory for more information.

## Data preparation
1. Download the dataset from the [`BOP benchmark`](https://bop.felk.cvut.cz/datasets/). Currently, our focus is on the LMO, TLESS, and YCBV datasets. We recommend using the LMO dataset for testing purposes due to its smaller size.
2. Download required ground truth (GT) folders of zebrapose from [`owncloud`](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW). The folders are `models_GT_color`, `XX_GT` (e.g. `train_pbr_GT` and `test_GT`) and `models` (`models` is optional, only if you want to generate GT from scratch, it contains more files needed to generate GT, but also contains all the origin files from BOP).

3. The expected data structure: 
    ```
    .
    └── BOP ROOT PATH/
        ├── lmo   
        ├── ycbv/
        │   ├── models            #(from step 1 or step 2, both OK)
        │   ├── models_eval
        │   ├── test              #(testing datasets)
        │   ├── train_pbr         #(training datasets)
        │   ├── train_real        #(not needed; we exclusively trained on PBR data.)
        │   ├── ...               #(other files from BOP page)
        │   ├── models_GT_color   #(from step 2)
        │   ├── train_pbr_GT      #(from step 2)
        │   ├── train_real_GT     #(from step 2)
        │   └── test_GT           #(from step 2)
        └── tless
    ```
4. (Optional) Instead of download the ground truth, you can also generate them from scratch, details in [`Generate_GT.md`](https://github.com/suyz526/ZebraPose/blob/main/Binary_Code_GT_Generator/Generate_GT.md).

## Testing
Download our trained model from this [`link`](https://1drv.ms/f/c/7b1c1126f255a9dd/Et2pVfImERwggHtwAAAAAAAB9jB0WOroeaU85GQqUK5EfA?e=og6KNj).
`python test.py --cfg config/test_lmo_config.txt --obj_name ape --ckpt_file /path/to/lmo/lmo_convnext_ape/0_7824step86000 --eval_output /path/to/eval_output --new_solver_version True --region_bit 10`

## Training
The script will save the last 3 checkpoints and the best checkpoint, as well as tensorboard log. 
Adjust the paths in the config files, and train the network with `train.py`, e.g.
`python train.py --cfg config/train_lmo_config.txt --obj_name ape`


The primary difference between `train_config.txt` and `test_config.txt` lies in the detection files they use. The provided checkpoints were trained using `train_config.txt`, and the results reported in the paper were obtained using `test_config.txt`. However, it should be perfectly acceptable to train using `test_config.txt` or to test using `train_config.txt`.

## Evaluate for BOP challange 
Merge the `.csv` files generated in the last step using `tools_for_BOP/merge_csv.py`, e.g.

`python merge_csv.py --input_dir /dir/to/pose_result_bop/lmo --output_fn hipose_lmo-test.csv`
We also provide our csv files from this [`link`](https://1drv.ms/f/s!At2pVfImERx7cM_BVybbo-ThTP4?e=wfbikU).

And then evaluate it according to [`bop_toolkit`](https://github.com/thodan/bop_toolkit).

## Acknowledgement
Some code are adapted from [`ZebraPose`](https://github.com/suyz526/ZebraPose), [`FFB6D`](https://github.com/ethnhe/FFB6D), [`Pix2Pose`](https://github.com/kirumang/Pix2Pose), [`SingleShotPose`](https://github.com/microsoft/singleshotpose), [`GDR-Net`](https://github.com/THU-DA-6D-Pose-Group/GDR-Net).
## Citation
```
@inproceedings{lin2024hipose,
  title={Hipose: Hierarchical binary surface encoding and correspondence pruning for rgb-d 6dof object pose estimation},
  author={Lin, Yongliang and Su, Yongzhi and Nathan, Praveen and Inuganti, Sandeep and Di, Yan and Sundermeyer, Martin and Manhardt, Fabian and Stricker, Didier and Rambach, Jason and Zhang, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10148--10158},
  year={2024}
}
```
