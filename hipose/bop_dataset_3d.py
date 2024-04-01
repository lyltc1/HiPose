import os

import torch
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

import sys
from binary_code_helper.class_id_encoder_decoder import RGB_image_to_class_id_image, class_id_image_to_class_code_images
import torchvision.transforms as transforms


import GDR_Net_Augmentation
from GDR_Net_Augmentation import get_affine_transform

import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
from scipy import interpolate

from depth_aug import add_noise_depth
from depth_aug import (
    DepthBlurTransform,
    DepthEllipseDropoutTransform,
    DepthGaussianNoiseTransform,
    DepthMissingTransform,
)

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0-x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1]-x1), (x2-x1))
    roi_y1 = max((0-y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0]-y1), (y2-y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size,crop_size), interpolation=interpolation)
    return roi_img

def crop_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = max(0, Bbox[0])
    x2 = min(img.shape[1], Bbox[0]+Bbox[2])
    y1 = max(0, Bbox[1])
    y2 = min(img.shape[0], Bbox[1]+Bbox[3])
    ####
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    ####

    img = img[y1:y2, x1:x2]
    roi_img = cv2.resize(img, (crop_size, crop_size), interpolation = interpolation)
    return roi_img

def get_scale_and_Bbox_center(Bbox, image):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    scale = max(bh, bw)
    scale = min(scale, max(image.shape[0], image.shape[1])) *1.0
    return scale, bbox_center

def get_roi(input, Bbox, crop_size, interpolation, resize_method):
    if resize_method == "crop_resize":
        roi = crop_resize(input, Bbox, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_resize_by_warp_affine":
        scale, bbox_center = get_scale_and_Bbox_center(Bbox, input)
        roi = crop_resize_by_warp_affine(input, bbox_center, scale, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_square_resize":
        roi = crop_square_resize(input, Bbox, crop_size, interpolation=interpolation)
        return roi
    else:
        raise NotImplementedError(f"unknown decoder type: {resize_method}")

def padding_Bbox(Bbox, padding_ratio):
    x1 = Bbox[0]
    x2 = Bbox[0] + Bbox[2]
    y1 = Bbox[1]
    y2 = Bbox[1] + Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    padded_bw = int(bw * padding_ratio)
    padded_bh = int(bh * padding_ratio)
        
    padded_Box = np.array([int(cx-padded_bw/2), int(cy-padded_bh/2), int(padded_bw), int(padded_bh)])
    return padded_Box

def aug_Bbox(GT_Bbox, padding_ratio):
    x1 = GT_Bbox[0].copy()
    x2 = GT_Bbox[0] + GT_Bbox[2]
    y1 = GT_Bbox[1].copy()
    y2 = GT_Bbox[1] + GT_Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
    # 1.5 is the additional pad scale
    augmented_bw = int(bw * scale_ratio * padding_ratio)
    augmented_bh = int(bh * scale_ratio * padding_ratio)
    
    augmented_Box = np.array([int(bbox_center[0]-augmented_bw/2), int(bbox_center[1]-augmented_bh/2), augmented_bw, augmented_bh])
    return augmented_Box

def get_final_Bbox(Bbox, resize_method, max_x, max_y):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh
    if resize_method == "crop_square_resize" or resize_method == "crop_resize_by_warp_affine":
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        if bh > bw:
            x1 = bbox_center[0] - bh/2
            x2 = bbox_center[0] + bh/2
        else:
            y1 = bbox_center[1] - bw/2
            y2 = bbox_center[1] + bw/2
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    elif resize_method == "crop_resize":
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, max_x)
        y2 = min(y2, max_y)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    return Bbox


def lin_interp(depth: np.ndarray) -> np.ndarray:
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid


def get_K_crop_resize(K, boxes, final_width, final_height):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape == (3, 3)
    assert boxes.shape == (4, )
    K = K.astype(float)
    boxes = boxes.astype(float)
    new_K = K.copy()

    crop_width = boxes[2] 
    crop_height = boxes[3]
    crop_cj = boxes[0] + boxes[2] /2
    crop_ci = boxes[1] + boxes[3] /2

    # Crop
    cx = K[0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[0, 0]
    fy = scale_y * K[1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[0, 0] = fx
    new_K[1, 1] = fy
    new_K[0, 2] = cx
    new_K[1, 2] = cy
    return new_K


class bop_dataset_single_obj_3d(Dataset):
    def __init__(self, dataset_dir, data_folder, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_params, 
                        is_train, crop_size_img, GT_code_infos, padding_ratio=1.5, resize_method="crop_resize", 
                        use_peper_salt=False, use_motion_blur=False, Detect_Bbox=None, sym_aware_training=False, dict_class_id_3D_points=None,
                        aug_depth=False, aug_depth_megapose6d = False, interpolate_depth=False, shift_center=False, shift_from_center=False):
        # gts: rotation and translation
        # gt_infos: bounding box
        self.rgb_files = rgb_files
        self.depth_files = depth_files
        self.mask_visib_files = mask_visib_files
        self.mask_files = mask_files
        self.gts = gts
        self.gt_infos = gt_infos
        self.cam_params = cam_params
        self.dataset_dir = dataset_dir
        self.data_folder = data_folder
        self.is_train = is_train
        self.GT_code_infos = GT_code_infos
        self.crop_size_img = crop_size_img
        self.resize_method = resize_method
        self.Detect_Bbox = Detect_Bbox
        self.padding_ratio = padding_ratio
        self.use_peper_salt = use_peper_salt
        self.use_motion_blur = use_motion_blur
        self.sym_aware_training = sym_aware_training
        self.dict_class_id_3D_points = dict_class_id_3D_points
        self.aug_depth = aug_depth
        self.aug_depth_megapose6d = aug_depth_megapose6d
        self.interpolate_depth = interpolate_depth
        self.shift_center= shift_center
        self.shift_from_center = shift_from_center 

        self.nSamples = len(self.rgb_files)
        
        if aug_depth_megapose6d:
            self.depth_augmentations = []
            self.depth_augmentations += [
                DepthBlurTransform(),
                DepthEllipseDropoutTransform(),
                DepthGaussianNoiseTransform(std_dev=0.01),
                DepthMissingTransform(max_missing_fraction=0.2),
            ]
        
    def __len__(self):
        return self.nSamples

    def Bbox_preprocessing(self, Bbox, index):
        if self.is_train:
            Bbox = aug_Bbox(Bbox, padding_ratio=self.padding_ratio)
        else:
            if self.Detect_Bbox!=None:
                # replace the Bbox with detected Bbox
                Bbox = self.Detect_Bbox[index]

                if Bbox is not None:
                    Bbox = padding_Bbox(Bbox, padding_ratio=self.padding_ratio)
        
        return Bbox


    def __getitem__(self, index):
        # return training image, mask, bounding box, R, T, GT_image
        rgb_fn = self.rgb_files[index]
        mask_visib_fns = self.mask_visib_files[index]
        mask_fns = self.mask_files[index]

        x = cv2.imread(rgb_fn)
        mask = cv2.imread(mask_visib_fns[0], 0)
        entire_mask = cv2.imread(mask_fns[0], 0)

        #rgb_files    ...datasetpath/train/scene_id/rgb/img.png
        rgb_fn = rgb_fn.split("/")
        scene_id = rgb_fn[-3]
        GT_image_name = mask_visib_fns[0].split("/")[-1]
        
        if self.sym_aware_training:
            GT_img_dir = os.path.join(self.dataset_dir, self.data_folder + '_GT_v2', scene_id)
        else:
            GT_img_dir = os.path.join(self.dataset_dir, self.data_folder + '_GT', scene_id)
        GT_img_fn = os.path.join(GT_img_dir, GT_image_name)        
        GT_img = cv2.imread(GT_img_fn)

        gt = self.gts[index]
        gt_info = self.gt_infos[index]

        if gt == None:  
            R = np.array(range(9)).reshape(3,3) 
            t = np.array(range(3)) 
            Bbox = np.array([1,1,1,1])
        else:
            R = np.array(gt['cam_R_m2c']).reshape(3,3) 
            t = np.array(gt['cam_t_m2c']) 
            Bbox = np.array(gt_info['bbox_visib'])

        cam_param = self.cam_params[index]['cam_K'].reshape((3,3))

        depth_image_mm = self.read_depth(self.depth_files[index], self.cam_params[index]["depth_scale"])
        #depth image augmentation
        if self.interpolate_depth:
            depth_image_mm = lin_interp(depth_image_mm / 1000.) * 1000.

        if self.is_train and self.aug_depth and not self.aug_depth_megapose6d:
            drop_ratio = 0.2
            drop_prob = 0.5
            if 'pbr' in self.data_folder and np.random.rand(1) < drop_prob: 
                keep_mask = np.random.uniform(0, 1, size=depth_image_mm.shape[:2])
                keep_mask = keep_mask > drop_ratio
                depth_image_mm = depth_image_mm * keep_mask

            add_noise_depth_prob = 0.3
            if np.random.rand(1) < add_noise_depth_prob:
                add_noise_depth_level = 0.01
                depth_image_mm = add_noise_depth(depth_image_mm, level=add_noise_depth_level)
            depth_image_m = depth_image_mm / 1000.
        elif self.is_train and not self.aug_depth and self.aug_depth_megapose6d:
            depth_image_m = depth_image_mm / 1000.
            for aug in self.depth_augmentations:
                if np.random.rand() < 0.3:
                    depth_image_m = aug(depth_image_m)
        else:
            depth_image_m = depth_image_mm / 1000.

        # crop all inputs
        Bbox = self.Bbox_preprocessing(Bbox, index)
        if not self.is_train and self.Detect_Bbox!=None:
            if Bbox is None: #no valid detection, give a dummy input
                inputs = {}
                targets = {}
                return inputs, targets

        Bbox = get_final_Bbox(Bbox, self.resize_method, x.shape[1], x.shape[0])
        cam_param_new = get_K_crop_resize(cam_param, Bbox, self.crop_size_img, self.crop_size_img)


        if self.is_train:        
            x = self.apply_augmentation(x)
        
        roi_rgb = get_roi(x, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR, resize_method = self.resize_method)
        roi_mask = get_roi(mask, Bbox, self.crop_size_img, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
        roi_entire_mask = get_roi(entire_mask, Bbox, self.crop_size_img, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
        roi_depth = get_roi(depth_image_m, Bbox, self.crop_size_img, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)

        roi_dpt_xyz = self.dpt_2_pcld(roi_depth, 1.0, cam_param_new, self.crop_size_img, self.crop_size_img)  # the second parameter is 1, so it not divide 1000 two times
        roi_dpt_xyz[np.isnan(roi_dpt_xyz)] = 0.0
        roi_dpt_xyz[np.isinf(roi_dpt_xyz)] = 0.0

        roi_depth_mm_int = (1000*roi_depth).astype(np.uint16)
        roi_nrm_map = normalSpeed.depth_normal(
            roi_depth_mm_int, cam_param_new[0,0], cam_param_new[1,1], 5, 2000, 20, False
        )

        original_center = np.zeros(3)
        if self.shift_center:
            # computer the center of points, and shift to 0
            original_center = np.mean(roi_dpt_xyz, axis=(0,1))
            original_center[2] = 0
            roi_dpt_xyz = roi_dpt_xyz - original_center
            if self.shift_from_center:
                shift_x = np.random.normal(0, 0.1)
                shift_y = np.random.normal(0, 0.1)
                roi_dpt_xyz[:,:,0] = roi_dpt_xyz[:,:,0] + shift_x
                roi_dpt_xyz[:,:,1] = roi_dpt_xyz[:,:,1] + shift_y
                original_center[0] = original_center[0] - shift_x
                original_center[1] = original_center[1] - shift_y

            

        visulize=False
        if visulize:
            show_nrm_map = ((roi_nrm_map + 1.0) * 127).astype(np.uint8)
            cv2.imwrite("/home/ysu/project/check_normal.png", show_nrm_map)

            dpt_xyz_full = self.dpt_2_pcld(depth_image_m, 1.0, cam_param, 640, 480)  # the second parameter is 1, so it not divide 1000 two times
            dpt_xyz_full[np.isnan(dpt_xyz_full)] = 0.0
            dpt_xyz_full[np.isinf(dpt_xyz_full)] = 0.0
            sys.path.append("../bop_toolkit")
            from bop_toolkit_lib import inout
            model = {}
            model["pts"] = dpt_xyz_full.reshape(-1,3)
            inout.save_ply("/home/ysu/project/check_pc_full.ply", model)
            model = {}
            model["pts"] = roi_dpt_xyz.reshape(-1,3)
            inout.save_ply("/home/ysu/project/check_pc.ply", model)
        
        mask_dp = roi_depth > 1e-6
        valid_depth_idx = mask_dp.flatten().nonzero()[0].astype(np.uint64)  # index of all valid points
        if len(valid_depth_idx) == 0:
            inputs = {}
            targets = {}
            return inputs, targets
            

        n_points = int(self.crop_size_img*self.crop_size_img/24)

        selected_idx = np.array([i for i in range(len(valid_depth_idx))])  # from 0 to length
        if len(selected_idx) > n_points:
            c_mask = np.zeros(len(selected_idx), dtype=int)
            c_mask[:n_points] = 1
            np.random.shuffle(c_mask)
            selected_idx = selected_idx[c_mask.nonzero()]  # if number of points are enough, random choose n_sample_points
        else:
            selected_idx = np.pad(selected_idx, (0, n_points-len(selected_idx)), 'wrap') 

        selected_point_idx = np.array(valid_depth_idx)[selected_idx]  # index of selected points, which has number of n_sample_points

        # shuffle the idx to have random permutation
        sf_idx = np.arange(selected_point_idx.shape[0])
        np.random.shuffle(sf_idx)
        selected_point_idx = selected_point_idx[sf_idx]   

        roi_cld = roi_dpt_xyz.reshape(-1, 3)[selected_point_idx, :]    # random selected points from all valid points
        if visulize:
            model = {}
            model["pts"] = roi_cld
            inout.save_ply("/home/ysu/project/check_selected_pc.ply", model)

        roi_pt_rgb = roi_rgb.reshape(-1, 3)[selected_point_idx, :].astype(np.float32)
        roi_pt_nrm = roi_nrm_map[:, :, :3].reshape(-1, 3)[selected_point_idx, :]

        selected_point_idx = np.array([selected_point_idx])
        roi_cld_rgb_nrm = np.concatenate((roi_cld, roi_pt_rgb, roi_pt_nrm), axis=1).transpose(1, 0)

        h = w = self.crop_size_img

        xyz_list = [roi_dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        mask_list = [roi_dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):   # add different scaled input into the list
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_list.append(xyz_list[0][:, ys*scale, xs*scale])    
            mask_list.append(xyz_list[-1][2, :, :] > 1e-8)

        scale2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_list)
        }     # c x h x w to h*w x 3 

        rgb_downsample_scale = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_sample_ratio = [4, 4, 4, 4]

        inputs = {}
        roi_cld_0 = roi_cld
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                roi_cld[None, ...], roi_cld[None, ...], 16
            ).astype(np.int64).squeeze(0)  # find 16 neiborhood for each point in the selected point cloud
            sub_pts = roi_cld[:roi_cld.shape[0] // pcld_sub_sample_ratio[i], :]    # can dowmsample , due to the index is schuffeled
            pool_i = nei_idx[:roi_cld.shape[0] // pcld_sub_sample_ratio[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], roi_cld[None, ...], 1
            ).astype(np.int64).squeeze(0)
            inputs['cld_xyz%d' % i] = roi_cld.astype(np.float32).copy()  # origin xyz
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int64).copy()  # find 16 neiborhood
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int64).copy()  # sub xyz neiborhood
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int64).copy()  # origin xyz find 1 neiborhoood in sub xyz
            nei_r2p = DP.knn_search(
                scale2dptxyz[rgb_downsample_scale[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int64).squeeze(0)  # sub xyz find 16 neiborhood in downsampled depth
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], scale2dptxyz[rgb_downsample_scale[i]][None, ...], 1
            ).astype(np.int64).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            roi_cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                scale2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int64).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                scale2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int64).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth = self.transform_pre_inputs(roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth)

        for key in inputs:
            inputs[key] = torch.from_numpy(inputs[key])

        inputs.update( 
            dict(
            rgb=roi_rgb,  # [c, h, w]
            cld_rgb_nrm=roi_cld_rgb_nrm,  # [9, npts]
            choose=selected_point_idx,  # [1, npts]
            dpt_map_m=roi_depth,  # [h, w]
            original_center = original_center
        )
        )

        if not os.path.exists(GT_img_fn):
            targets = {}
            return inputs, targets

        # start to process the training labels
        obj_pts_mask = roi_mask.flatten()[selected_point_idx.flatten()]/255 # which point belongs to the object
        
        # transform the model points into the corresponded pose, and find NN of obj pts and point on the model
        # load the model points
        roi_GT_img = get_roi(GT_img, Bbox, self.crop_size_img, interpolation=cv2.INTER_NEAREST, resize_method = self.resize_method)
        class_id_image= RGB_image_to_class_id_image(roi_GT_img)

        obj_pts_id = class_id_image.flatten()[selected_point_idx.flatten()]
        
        # check distance to filter noisy depth measurement
        model_pts = np.zeros((obj_pts_id.shape[0], 3))

        for i, id in enumerate(obj_pts_id):
            model_pts[i] = self.dict_class_id_3D_points[id]  # in mm
        
        model_pts_transformed = (R @ model_pts.transpose() + t) / 1000. # in m
        
        if self.shift_center:
            observed_cloud = roi_cld_0 + original_center
        else:
            observed_cloud = roi_cld_0

        dist = np.linalg.norm(model_pts_transformed.transpose() - observed_cloud, axis=1) 

        obj_pts_code = class_id_image_to_class_code_images(obj_pts_id.reshape(-1,1), self.GT_code_infos[0], self.GT_code_infos[1], self.GT_code_infos[2])

        obj_pts_code, obj_pts_mask = self.transform_pre_targets(obj_pts_code, obj_pts_mask)
        targets = {'obj_pts_mask': obj_pts_mask,
                   'obj_pts_code': obj_pts_code,
                   'Rs': R,
                   'ts': t
                   }
        return inputs, targets
       


    def visulize(self, x, entire_mask, mask, GT_img_visible, GT_img_invisible, Bbox):
        cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('entire_mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('GT_img_visible', cv2.WINDOW_NORMAL)
        cv2.namedWindow('GT_img_invisible', cv2.WINDOW_NORMAL)

        x_ = x.copy()
        if Bbox is not None:
            cv2.rectangle(x_,(Bbox[0],Bbox[1]),(Bbox[0]+Bbox[2] ,Bbox[1]+Bbox[3] ),(0,255,0),3) 
        cv2.imshow('rgb',x_)
        cv2.imshow('mask',mask)
        cv2.imshow('entire_mask',entire_mask)
        
        cv2.imshow('GT_img_visible',GT_img_visible)
        cv2.imshow('GT_img_invisible',GT_img_invisible)

        cv2.waitKey(0)

    @staticmethod
    def transform_pre_inputs(roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth):
        composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        roi_rgb = Image.fromarray(np.uint8(roi_rgb)).convert('RGB')
        
        roi_cld_rgb_nrm = roi_cld_rgb_nrm.astype(np.float32)
        roi_cld_rgb_nrm = torch.from_numpy(roi_cld_rgb_nrm)
        
        selected_point_idx = selected_point_idx.astype(np.int64)
        selected_point_idx = torch.from_numpy(selected_point_idx)
        
        roi_depth = roi_depth.astype(np.float32)
        roi_depth = torch.from_numpy(roi_depth)
        return composed_transforms_img(roi_rgb), roi_cld_rgb_nrm, selected_point_idx, roi_depth
    
    def transform_pre_targets(self, obj_pts_code, obj_pts_mask):
        obj_pts_code = torch.from_numpy(obj_pts_code).permute(2, 0, 1) 
        
        obj_pts_mask = torch.from_numpy(obj_pts_mask).type(torch.float)
        return obj_pts_code, obj_pts_mask

    def apply_augmentation(self, x):
        augmentations = GDR_Net_Augmentation.build_augmentations(self.use_peper_salt, self.use_motion_blur)      
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = augmentations.augment_image(x)

        return x
    
    @staticmethod
    def read_depth(path, scale):
        depth = np.asarray(Image.open(path)).copy()
        depth = depth.astype(np.float32)
        depth = depth * scale
        return depth

    @staticmethod
    def dpt_2_pcld(dpt, cam_scale, K, img_w, img_h):
        xmap = np.array([[j for i in range(img_w)] for j in range(img_h)])
        ymap = np.array([[i for i in range(img_w)] for j in range(img_h)])
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (ymap - K[0,2]) * dpt / K[0,0]
        col = (xmap - K[1,2]) * dpt / K[1,1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d


if __name__ == "__main__":
    from tools_for_BOP import bop_io
    from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points
    bop_path = '/home/ysu/data/data_object_pose/BOP'
    dataset_name = 'lmo'
    training_data_folder = 'train_real'
    train_obj_visible_theshold = 0.2
    obj_id = 8

    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(2)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(16)

    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]
    
    dataset = bop_dataset_single_obj_3d(dataset_dir, training_data_folder, rgb_files[obj_id], depth_files[obj_id], mask_files[obj_id], mask_visib_files[obj_id], 
                                                    gts[obj_id], gt_infos[obj_id], cam_params[obj_id], True, 256, 
                                                    GT_code_infos,  padding_ratio=1.5, resize_method="crop_square_resize", 
                                                    use_peper_salt=True, use_motion_blur=True, sym_aware_training=False, dict_class_id_3D_points=dict_class_id_3D_points, 
                                                    aug_depth=False, interpolate_depth=False, shift_center=True 
                                                    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    train_loader_iter = iter(train_loader)
    
    data = next(train_loader_iter)
    