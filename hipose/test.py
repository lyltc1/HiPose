from distutils.command.config import config
import os
import sys

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse
import shutil
from tqdm import tqdm


from tools_for_BOP import bop_io

import torch
import numpy as np
import cv2

from binary_code_helper.CNN_output_to_pose_region import CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v5
from binary_code_helper.CNN_output_to_pose_region_for_test_with_region_v3 import CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7
sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout

from itertools import groupby


from models.ffb6d import FFB6D
from models.common import ConfigRandLA

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, compute_original_mask
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.CNN_output_to_pose_region import load_dict_class_id_3D_points
from binary_code_helper.generate_new_dict import generate_new_corres_dict_and_region

import cv2

from tools_for_BOP import write_to_cvs 

import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
import json

import time

import pickle


def prepare_input_dict(dataset_dir_test, camera_params_dict, Detected_Bbox, resize_method, padding_ratio, BoundingBox_CropSize_image, related_functions, rgb_fn):
    bop_dataset_single_obj_3d, get_K_crop_resize, padding_Bbox, get_final_Bbox, get_roi = related_functions

    rgb_fn_splitted = rgb_fn.split("/")
    scene_id = rgb_fn_splitted[-3]
    img_id = rgb_fn_splitted[-1].split(".")[0]
    rgb_fname = rgb_fn

    depth_fn = os.path.join(dataset_dir_test,scene_id,'depth',"{}.png".format(img_id))

    #Cam_K = camera_params_test[obj_id][0]['cam_K'].reshape((3,3))
    Cam_K = camera_params_dict[scene_id][int(img_id)]['cam_K'].reshape((3,3))

    rgb_img = cv2.imread(rgb_fname)

    original_Bbox = Detected_Bbox['bbox_est']
    Bbox = padding_Bbox(original_Bbox, padding_ratio=padding_ratio)
    roi_rgb = get_roi(rgb_img, Bbox, BoundingBox_CropSize_image, interpolation=cv2.INTER_LINEAR, resize_method=resize_method)
    Bbox = get_final_Bbox(Bbox, resize_method, rgb_img.shape[1], rgb_img.shape[0])
    # cv2.imwrite("rgb.png", roi_rgb)
    depth_image_mm = bop_dataset_single_obj_3d.read_depth(depth_fn, camera_params_dict[scene_id][int(img_id)]["depth_scale"])
    depth_image_m = depth_image_mm / 1000.
    cam_param_new = get_K_crop_resize(Cam_K, Bbox, BoundingBox_CropSize_image, BoundingBox_CropSize_image)
    roi_depth = get_roi(depth_image_m, Bbox, BoundingBox_CropSize_image, interpolation=cv2.INTER_NEAREST, resize_method = resize_method)

    roi_dpt_xyz = bop_dataset_single_obj_3d.dpt_2_pcld(roi_depth, 1.0, cam_param_new, BoundingBox_CropSize_image,BoundingBox_CropSize_image)  # the second parameter is 1, so it not divide 1000 two times
    roi_dpt_xyz[np.isnan(roi_dpt_xyz)] = 0.0
    roi_dpt_xyz[np.isinf(roi_dpt_xyz)] = 0.0

    roi_depth_mm_int = (1000*roi_depth).astype(np.uint16)
    roi_nrm_map = normalSpeed.depth_normal(
        roi_depth_mm_int, cam_param_new[0,0], cam_param_new[1,1], 5, 2000, 20, False
    )

    mask_dp = roi_depth > 1e-6
    valid_depth_idx = mask_dp.flatten().nonzero()[0].astype(np.uint64)  # index of all valid points
    if len(valid_depth_idx) == 0:
        return 0,0,{}

    n_points = int(BoundingBox_CropSize_image*BoundingBox_CropSize_image/24)

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

    roi_pt_rgb = roi_rgb.reshape(-1, 3)[selected_point_idx, :].astype(np.float32)
    roi_pt_nrm = roi_nrm_map[:, :, :3].reshape(-1, 3)[selected_point_idx, :]

    selected_point_idx = np.array([selected_point_idx])
    roi_cld_rgb_nrm = np.concatenate((roi_cld, roi_pt_rgb, roi_pt_nrm), axis=1).transpose(1, 0)

    h = w = BoundingBox_CropSize_image

    xyz_list = [roi_dpt_xyz.transpose(2, 0, 1)]  # c, h, w
    mask_list = [roi_dpt_xyz[2, :, :] > 1e-8]

    for i in range(4):   # add different scaled input into the list
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
    rgb_downsample_scale = [4, 8, 16, 16]
    n_ds_layers = 4
    pcld_sub_sample_ratio = [4, 4, 4, 4]

    inputs = {}
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
    rgb_up_sr = [8, 4, 4]
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

    roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth = bop_dataset_single_obj_3d.transform_pre_inputs(roi_rgb, roi_cld_rgb_nrm, selected_point_idx, roi_depth)

    for key in inputs:
        inputs[key] = torch.from_numpy(inputs[key])

    inputs.update( 
        dict(
        rgb=roi_rgb,  # [c, h, w]
        cld_rgb_nrm=roi_cld_rgb_nrm,  # [9, npts]
        choose=selected_point_idx,  # [1, npts]
        dpt_map_m=roi_depth,  # [h, w]
        Bbox = Bbox
    )
    )

    inputs["cam_param_new"] = cam_param_new

    return scene_id, img_id, inputs


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap


def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    test_folder=configs['test_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_theshold = configs['train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    convnext = configs.get('convnext', False)
    if convnext is False:
        from bop_dataset_3d import bop_dataset_single_obj_3d, get_K_crop_resize, padding_Bbox, get_final_Bbox, get_roi
    elif convnext.startswith('convnext'):
        from bop_dataset_3d_convnext_backbone import bop_dataset_single_obj_3d, get_K_crop_resize, padding_Bbox, get_final_Bbox, get_roi
    related_functions = (bop_dataset_single_obj_3d, get_K_crop_resize, padding_Bbox, get_final_Bbox, get_roi)

    fusion = configs.get('fusion', False)
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"          

    #### augmentations
    Detection_reaults=configs['Detection_reaults']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256

    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']


    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments

    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1) # now the obj_id started from 0

    mesh_path = model_plys[obj_id+1] # mesh_path is a dict, the obj_id should start from 1
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)

    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    vertices = inout.load_ply(mesh_path)["pts"]

    # define test data loader
    if False:
        dataset_dir_test,bop_test_folder,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name,train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, train_obj_visible_theshold=train_obj_visible_theshold)
    else:
        print("use BOP test images")
        dataset_dir_test,bop_test_folder,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder)


    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length

    rndla_cfg = ConfigRandLA
    net = FFB6D(
        n_classes=1, n_pts=480 * 640 // 24 , rndla_cfg=rndla_cfg,
        number_of_outputs=binary_code_length + 1, fusion=fusion,
        convnext=convnext
    )

    if torch.cuda.is_available():
        net=net.cuda()

    checkpoint = torch.load( configs['checkpoint_file'] )
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    bit2class_id_center_and_region = {}
    for bit in range(configs['region_bit'] + 1, 17):
        bit2class_id_center_and_region[bit] = generate_new_corres_dict_and_region(dict_class_id_3D_points, 16, bit)

    # complete bit2class_id_center_and_region so that all regions share the same shape, default: 32
    region_max_points = pow(2, 15 - configs['region_bit'])
    for bit in range(configs['region_bit'] + 1, 17):
        for center_and_region in bit2class_id_center_and_region[bit].values():
            region = center_and_region['region']
            assert region.shape[0] <= region_max_points
            if region.shape[0] < region_max_points:
                region_new = np.zeros([region_max_points, 3])
                region_new[:region.shape[0]] = region
                region_new[region.shape[0]:] = region[0]
                center_and_region['region'] = region_new

    img_ids = []
    scene_ids = []
    estimated_Rs = []
    estimated_Ts = []
    scores = []
    times = []

    test_rgb_files_no_duplicate = list(dict.fromkeys(test_rgb_files[obj_id]))

    if configs['detector']=='FCOS':
        from get_detection_results import get_detection_results_vivo
    elif configs['detector']=='MASKRCNN':
        from get_mask_rcnn_results import get_detection_results_vivo
    Bboxes = get_detection_results_vivo(Detection_reaults, test_rgb_files_no_duplicate, obj_id+1, 0.5)

    ##get camera parameters
    camera_params_dict = dict()
    for scene_id in os.listdir(bop_test_folder):
        current_dir = bop_test_folder+"/"+scene_id
        scene_params = inout.load_scene_camera(os.path.join(current_dir,"scene_camera.json"))     
        camera_params_dict[scene_id] = scene_params    

    for rgb_fn, Bboxes_frame in tqdm(Bboxes.items()):
        
        for Detected_Bbox in Bboxes_frame:
            start_time = time.time()
            scene_id, img_id, inputs = prepare_input_dict(bop_test_folder, camera_params_dict, Detected_Bbox, resize_method, padding_ratio, BoundingBox_CropSize_image, related_functions, rgb_fn)

            if not inputs: # no valid detected bbox
                continue
            if torch.cuda.is_available():
                for key in inputs:
                    if not isinstance(inputs[key], torch.Tensor):
                        inputs[key] = torch.tensor(inputs[key])
                    inputs[key] = torch.unsqueeze(inputs[key], 0).cuda()

            pred_masks_prob, pred_code_prob = net(inputs)
            pred_masks_probability = torch.sigmoid(pred_masks_prob).detach().cpu().numpy()
            pred_codes_probability = torch.sigmoid(pred_code_prob).detach().cpu().numpy()

            inputs_pc = inputs['cld_xyz0'].detach().cpu().numpy()
            
            if configs["new_solver_version"] == False:
                R_predict, t_predict, success = CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v5(
                    inputs_pc[0],
                    pred_masks_probability[0], 
                    pred_codes_probability[0], 
                    bit2class_id_center_and_region=bit2class_id_center_and_region,
                    dict_class_id_3D_points=dict_class_id_3D_points) 
            else:
                R_predict, t_predict, success = CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7(
                    inputs_pc[0], 
                    pred_masks_probability[0], 
                    pred_codes_probability[0], 
                    bit2class_id_center_and_region=bit2class_id_center_and_region,
                    dict_class_id_3D_points=dict_class_id_3D_points,
                    region_bit=configs["region_bit"],
                    threshold=50,
                    mean=False,
                    uncertain_threshold = 0.02
                    ) 

            end_time = time.time()
            ##################
            score = Detected_Bbox['score']

            if success:     
                img_ids.append(img_id)
                scene_ids.append(scene_id) 
                estimated_Rs.append(R_predict)
                estimated_Ts.append(t_predict)
                scores.append(score)
                times.append(end_time-start_time)

    cvs_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)
    write_to_cvs.write_cvs(cvs_path, "{}_{}".format(dataset_name, obj_name), obj_id+1, scene_ids, img_ids, estimated_Rs, estimated_Ts, scores, times)

    print("csv path", cvs_path)

    record_run_time_path = os.path.join(eval_output_path, 'runtime')
    if not os.path.exists(record_run_time_path):
        os.makedirs(record_run_time_path)
    record_run_time_file = os.path.join(record_run_time_path, '{}.txt'.format(obj_name))                 
    with open(record_run_time_file, 'w') as file:
        file.write(str(np.array(times).mean()))

    # print("avg. time", np.array(times).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--region_bit', type=int, choices=range(16), default=10, help="the bit index")
    parser.add_argument('--new_solver_version', type=str, choices=('True','False'), default='False')

    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    eval_output_path = args.eval_output_path
    obj_name = args.obj_name
    configs = parse_cfg(config_file)

    configs['obj_name'] = obj_name
    configs['new_solver_version'] = (args.new_solver_version == 'True')
    configs['region_bit'] = args.region_bit


    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = eval_output_path

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
