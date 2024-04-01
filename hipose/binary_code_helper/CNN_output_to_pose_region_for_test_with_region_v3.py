from binary_code_helper.class_id_encoder_decoder import class_code_images_to_class_id_image
import random
import os
import numpy as np
import torch
import cv2
import open3d as o3d

def load_dict_class_id_3D_points(path):
    total_numer_class = 0
    number_of_itration = 0

    dict_class_id_3D_points = {}
    with open(path, "r") as f:
        first_line = f.readline()
        total_numer_class_, divide_number_each_itration, number_of_itration_ = first_line.split(" ") 
        divide_number_each_itration = float(divide_number_each_itration)
        total_numer_class = float(total_numer_class_)
        number_of_itration = float(number_of_itration_)

        for line in f:
            line = line[:-1]
            code, x, y, z= line.split(" ")
            code = float(code)
            x = float(x)
            y = float(y)
            z = float(z)

            dict_class_id_3D_points[code] = np.array([x,y,z])

    return total_numer_class, divide_number_each_itration, number_of_itration, dict_class_id_3D_points


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return R, t


def RANSAC_best_fit_v2(A, B, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30):
    max_number_inliers = 0
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    # print(A.shape)  N x 3
    lens_samples = len(A)

    if lens_samples > num_sub_sample_pts:
        for i in range(max_num_itrations):
            index = np.random.randint(0, lens_samples, num_sub_sample_pts)

            sub_A = A[index]
            sub_B = B[index]
            try:
                R, t = best_fit_transform(sub_A, sub_B)
            except:
                continue

            transformed_A = R @ A.transpose()  + t.reshape(3,1)

            dist = np.linalg.norm(transformed_A.transpose() - B, axis=1) 
            inlier = A[dist < inlier_threshold]  

            n_inlier = len(inlier)
         

            if n_inlier > max_number_inliers:
                max_number_inliers = n_inlier
                # solve R and t with all inliers
                inlier_A = A[dist < 3]  
                inlier_B = B[dist < 3]  
                try:
                    R, t = best_fit_transform(inlier_A, inlier_B)
                    best_rot = R
                    best_tvecs = t
                except:    
                    best_rot = R
                    best_tvecs = t
    
    else:
        try:
            best_rot, best_tvecs = best_fit_transform(A, B)
        except:
            return best_rot, best_tvecs
    #print(max_number_inliers)
    return best_rot, best_tvecs


def RANSAC_best_fit_open3d(A, B, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30):
    max_number_inliers = 0
    most_inliers_index = []
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    # print(A.shape)  N x 3
    lens_samples = len(A)

    solver = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(A)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(B)
    if lens_samples > num_sub_sample_pts:
        for i in range(max_num_itrations):
            index = random.sample(range(lens_samples), num_sub_sample_pts)
            known_correspondences = [[i, i] for i in index]
            known_correspondences = o3d.utility.Vector2iVector(known_correspondences)
            try:
                result = solver.compute_transformation(source_pcd, target_pcd, known_correspondences)
                R = result[0:3, :3]
                t = result[0:3, 3]
            except:
                continue

            transformed_A = R @ A.transpose()  + t.reshape(3,1)

            dist = np.linalg.norm(transformed_A.transpose() - B, axis=1) 
            n_inlier = np.sum(dist < inlier_threshold)
          
            if n_inlier > max_number_inliers:
                most_inliers_index = dist < inlier_threshold
                max_number_inliers = n_inlier

        # solve R and t with all inliers
        try:
            known_correspondences = [[i, i] for i, k in enumerate(most_inliers_index) if k]
            known_correspondences = o3d.utility.Vector2iVector(known_correspondences)

            result = solver.compute_transformation(source_pcd, target_pcd, known_correspondences)
            R = result[0:3, :3]
            t = result[0:3, [3]]
            best_rot = R
            best_tvecs = t
        except:    
            best_rot = R
            best_tvecs = t
    
    else:
        try:
            known_correspondences = []
            for indx in range(len(source_pcd.points)):
                known_correspondences.append([indx,indx])
            known_correspondences = o3d.utility.Vector2iVector(known_correspondences)

            result = solver.compute_transformation(source_pcd, target_pcd, known_correspondences)
            best_rot = result[0:3, :3]
            best_tvecs = result[0:3, [3]]
        except:
            return best_rot, best_tvecs
    #print(max_number_inliers)
    return best_rot, best_tvecs


def RANSAC_best_fit_v5(A, B, max_num_itrations=10, inlier_threshold=3, num_sub_sample_pts=20):
    max_number_inliers = 0
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    lens_samples = len(A)

    if lens_samples > num_sub_sample_pts:
        for i in range(max_num_itrations):
            lens_samples = len(A)
            if lens_samples < num_sub_sample_pts:
                continue
            index = random.sample(range(lens_samples), num_sub_sample_pts)

            sub_A = A[index]
            sub_B = B[index]
            try:
                R, t = best_fit_transform(sub_A, sub_B)
            except:
                continue

            transformed_A = R @ A.transpose()  + t.reshape(3,1)

            dist = np.linalg.norm(transformed_A.transpose() - B, axis=1)
            inlier = A[dist < inlier_threshold]  

            n_inlier = len(inlier)
         
            if n_inlier > max_number_inliers:
                max_number_inliers = n_inlier
                # solve R and t with all inliers
                inlier_A = A[dist < 3]
                inlier_B = B[dist < 3]
                try:
                    R, t = best_fit_transform(inlier_A, inlier_B)
                    best_rot = R
                    best_tvecs = t
                except:
                    best_rot = R
                    best_tvecs = t
                keep_index = np.argsort(dist)[:int(lens_samples * 0.9)]
                A = A[keep_index]
                B = B[keep_index]
    else:
        try:
            best_rot, best_tvecs = best_fit_transform(A, B)
        except:
            return best_rot, best_tvecs
    #print(max_number_inliers)
    return best_rot, best_tvecs


def RANSAC_best_fit_region(
        A,  # model points [npts, 3]
        B,  # observed points [npts, 3]
        chosen_observed_pts,  # a list contains n_chosen array whose shape is [3,]
        chosen_obj_model_regions,  # a list contains n_chosen array whose shape is [n_pts_in_a_region, 3]
        max_num_itrations=300, 
        inlier_threshold=3, 
        num_sub_sample_pts=30):
    max_number_inliers = 0
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    # print(A.shape)  N x 3
    lens_samples = len(A)

    if lens_samples > num_sub_sample_pts:
        for i in range(max_num_itrations):
            index = random.sample(range(lens_samples), num_sub_sample_pts)

            sub_A = A[index]
            sub_B = B[index]
            try:
                R, t = best_fit_transform(sub_A, sub_B)
            except:
                continue

            n_inlier_for_region = 0 
            for pts, regions in zip(chosen_observed_pts, chosen_obj_model_regions):
                transformed_regions = R @ regions.transpose() + t.reshape(3,1)
                dist_in_region = np.linalg.norm(transformed_regions.transpose() - pts[None], axis=1) 
                if min(dist_in_region) < inlier_threshold:
                    n_inlier_for_region += 1
         

            if n_inlier_for_region > max_number_inliers:
                max_number_inliers = n_inlier_for_region
                # solve R and t with all inliers
                transformed_A = R @ A.transpose()  + t.reshape(3,1)
                dist = np.linalg.norm(transformed_A.transpose() - B, axis=1) 

                inlier_A = A[dist < 3]
                inlier_B = B[dist < 3]
                try:
                    R, t = best_fit_transform(inlier_A, inlier_B)
                    best_rot = R
                    best_tvecs = t
                except:    
                    best_rot = R
                    best_tvecs = t
    else:
        try:
            best_rot, best_tvecs = best_fit_transform(A, B)
        except:
            return best_rot, best_tvecs
    return best_rot, best_tvecs


def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v1(
        input_cloud, 
        mask_probability, 
        code_probability, 
        dict_class_id_3D_points=None, 
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None, 
        t_gt=None, 
        ):
    """
    caculate initial Rt, and refine based on region, version 1
    input_cloud: [npts, 3]
    mask_probability: [npts, 1], is a probability range from [0, 1]
    code_probability: [npts, 16], is a probability range from [0, 1]
    """

    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False
    
    # # generate gt_obj_model_pts from class_code_image_target, for visualization only
    # gt_code_image = class_code_image_target[:, 0, :][obj_pts_idx]
    # gt_class_id = class_code_images_to_class_id_image(gt_code_image[None], class_base)[0]
    # gt_obj_model_pts = np.zeros((gt_class_id.shape[0], 3))
    # for i, id in enumerate(gt_class_id):
    #     gt_obj_model_pts[i] = dict_class_id_3D_points[id]

    input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]

    obj_model_pts = np.zeros((class_id.shape[0], 3))  # [npts, 3]
    for i, id in enumerate(class_id):
        obj_model_pts[i] = dict_class_id_3D_points[id]
    rot, tvecs = RANSAC_best_fit_v5(obj_model_pts, observed_pts, max_num_itrations=10, inlier_threshold=3, num_sub_sample_pts=20)

    for bit in [10, ]:
        code_image_floor = np.zeros_like(code_image, dtype=np.uint32)
        code_image_floor[..., :bit] = code_image[..., :bit]
        class_id_floor = class_code_images_to_class_id_image(code_image_floor[None], class_base)[0]
        code_image_ceil = np.ones_like(code_image, dtype=np.uint32)
        code_image_ceil[..., :bit] = code_image[..., :bit]
        class_id_ceil = class_code_images_to_class_id_image(code_image_ceil[None], class_base)[0]
        obj_model_regions = []
        
        for id_floor, id_ceil in zip(class_id_floor, class_id_ceil):
            obj_model_region = []
            for id in range(id_floor, id_ceil + 1):
                obj_model_region.append(dict_class_id_3D_points[id])
            obj_model_regions.append(np.stack(obj_model_region))
        obj_model_regions = np.stack(obj_model_regions)  # [npts, n_region_points, 3]
        transformed_obj_model_regions = obj_model_regions @ rot.T[None] + tvecs.reshape((1, 1, 3))
        dist = np.linalg.norm(observed_pts[:, None] - transformed_obj_model_regions, axis=2)
        idx = np.argmin(dist, axis=1)

        dist_min = np.min(dist, axis=1)
        obj_model_nearest = obj_model_regions[np.arange(num), idx]
        rot, tvecs = RANSAC_best_fit_v5(
            obj_model_nearest[dist_min < np.median(dist_min)], 
            observed_pts[dist_min < np.median(dist_min)], 
            max_num_itrations=1, 
            inlier_threshold=3, 
            num_sub_sample_pts=20)

    return rot, tvecs.reshape((3,1)), True

def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v2(
        input_cloud, 
        mask_probability, 
        code_probability, 
        dict_class_id_3D_points=None, 
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None, 
        t_gt=None, 
        ):
    """
    input_cloud: [npts, 3]
    mask_probability: [npts, 1], is a probability range from [0, 1]
    code_probability: [npts, 16], is a probability range from [0, 1]
    """

    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False
    
    # # generate gt_obj_model_pts from class_code_image_target, for visualization only
    # gt_code_image = class_code_image_target[:, 0, :][obj_pts_idx]
    # gt_class_id = class_code_images_to_class_id_image(gt_code_image[None], class_base)[0]
    # gt_obj_model_pts = np.zeros((gt_class_id.shape[0], 3))
    # for i, id in enumerate(gt_class_id):
    #     gt_obj_model_pts[i] = dict_class_id_3D_points[id]

    input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]

    codes_length = code_probability.shape[1]
    obj_model_pts = np.zeros((class_id.shape[0], 3))  # [npts, 3]
    chosen_observed_pts = []
    chosen_obj_model_regions = []
    for i, (id, prob) in enumerate(zip(class_id, code_probability)):
        obj_model_pts[i] = dict_class_id_3D_points[id]

        class_id_min = 0
        class_id_max = None
        for j in range(16):
            if prob[j] > 0.52:
                class_id_min += class_base**(codes_length - 1 - j)
            elif prob[j] < 0.48:
                pass
            else:
                class_id_max = class_id_min
                for k in range(j, 16):
                    class_id_max += class_base**(codes_length - 1 - k)
                break
        if class_id_max is None:
            class_id_max = class_id_min

        if class_id_max - class_id_min <= 64:  # update chosen pts and regions
            region = []
            for c in range(class_id_min, class_id_max + 1):
                region.append(dict_class_id_3D_points[c])
            chosen_obj_model_regions.append(np.stack(region))
            chosen_observed_pts.append(observed_pts[i])
    
    rot, tvecs = RANSAC_best_fit_region(
        obj_model_pts, 
        observed_pts,
        chosen_observed_pts,
        chosen_obj_model_regions,
        max_num_itrations=10, 
        inlier_threshold=3, 
        num_sub_sample_pts=20)
    success = True
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success


def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v5(
        input_cloud,
        mask_probability,
        code_probability,
        bit2class_id_center_and_region,
        dict_class_id_3D_points,
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None,
        t_gt=None,
        rgb=None,
        depth=None,
        K=None,
        debug_image_dir=None,
        renderer=None,
        ):
    """
    accerate script as v3
    """
    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]

    # prepare score for each correspondences
    score = np.zeros((6, num, 1))  # [n_bit, n_pts, 1]
    for i, bit in enumerate([10, 11, 12, 13, 14, 15]):
        score[i] = (0.5 - code_probability[:, [bit,]]) ** 2

    # prepare corresponding region and center for different bits
    bits = [10, 11, 12, 13, 14, 15]
    obj_model_region = {bit:[] for bit in bits}
    obj_model_region_center = {bit:[] for bit in bits}
    for bit in bits:
        clip_class_id = class_id >> (16 - bit)
        for c_id in clip_class_id:
            region = bit2class_id_center_and_region[bit][c_id]['region']
            obj_model_region[bit].append(region)
            obj_model_region_center[bit].append(bit2class_id_center_and_region[bit][c_id]['center'])
        obj_model_region_center[bit] = np.stack(obj_model_region_center[bit])
        obj_model_region[bit] = np.stack(obj_model_region[bit])

    VISUALIZE = False
    if VISUALIZE:
        mean = torch.as_tensor((0.485, 0.456, 0.406), device=rgb.device)
        std = torch.as_tensor((0.229, 0.224, 0.225), device=rgb.device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        x = rgb.mul(std).add(mean)
        x = x.detach().cpu().numpy()
        x = (x.transpose(1, 2, 0)*255.).astype(np.uint16)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        gt_pose, _ = renderer.render(0, 256, 256, K, R_gt, t_gt / 1000., 0.25, 6.0)
        depth = depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.
        depth = depth.astype(np.uint16)
        depth_show = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        depth_show[:,:,0] = depth
        depth_show[:,:,1] = depth
        depth_show[:,:,2] = depth
        show_ims = [x, gt_pose[:, :, [2, 1, 0]], depth_show]
        show_titles = ["image", "gt_pose", "depth"]
        grid_show(show_ims, show_titles, row=1, col=3, save_path=os.path.join(debug_image_dir, "img_and_gt.jpg"))

                    
    rot, tvecs = best_fit_region_center(
        obj_model_region_center, 
        observed_pts,
        obj_model_region,
        bits = bits,
        score = score,
        inlier_threshold=3, 
        )
    success = True
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success


def best_fit_region_center(
        obj_model_region_center,
        observed_pts,
        obj_model_region,
        bits,
        score=None,
        inlier_threshold=3,
        ):
    R = np.zeros((3,3))
    t = np.zeros((3,1))

    keep = np.ones(len(observed_pts), dtype=bool)
    keeps = []
    for i, bit in enumerate(bits):
        region_center = obj_model_region_center[bit]  # [npts, 3]
        region = obj_model_region[bit]         # [npts, n_pts_in_a_region, 3]
        R, t = best_fit_transform(region_center[keep], observed_pts[keep])

        transformed_region = R[None, None] @ region[..., None] + t.reshape(3,1)[None, None, ...]
        dist = np.linalg.norm(transformed_region[:,:,:,0] - observed_pts[:, None, :], axis=-1)
        min_dist = np.min(dist, axis=1)  # [npts, ]
        # min_dist = min_dist * (1-score[i,:,0])
        keep = min_dist < np.median(min_dist)
        keeps.append(keep.copy())
    keep_ = np.ones(len(observed_pts), dtype=bool)
    for i in range(len(keeps)):
        keep_ = np.logical_and(keep_, keeps[i])
    print("final num points", keep_.sum())
    R, t = RANSAC_best_fit_v2(region_center[keep_], observed_pts[keep_],max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30)
    # R, t = best_fit_transform(region_center[keep_], observed_pts[keep_])
    return R, t


def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v6(
        input_cloud,
        mask_probability,
        code_probability,
        bit2class_id_center_and_region,
        dict_class_id_3D_points,
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None,
        t_gt=None,
        rgb=None,
        depth=None,
        K=None,
        debug_image_dir=None,
        renderer=None,
        ):
    """
    The region bit is decided on probability in v6, 
    also we will divide observed_pts into different group based on region bit.
    """
    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    # input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]

    # prepare corresponding region and center for different correspondence, set first n bits whose number is not in [0.48, 0.52] as init region.
    iter_num = 7
    obj_model_region_center = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [3,]
    obj_model_region = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [varied_, 3]
    obj_model_region_bit_index = []  # a list whose num is n, the last index in bits which is outside [0.48, 0.52]

    for c_id, prob in zip(class_id, code_probability):
        index = 10
        for j in range(10, 16):
            if 0.48 < prob[j] < 0.52:
                break
            else:
                index = j
        obj_model_region_bit_index.append(index)
        for iter in range(0, iter_num):
            region_class_id = c_id >> max(15 - index - iter, 0)
            bit = 16 - max(15 - index - iter, 0)
            region = bit2class_id_center_and_region[bit][region_class_id]['region']
            obj_model_region[iter].append(region)
            center = bit2class_id_center_and_region[bit][region_class_id]['center']
            obj_model_region_center[iter].append(center)

    rot, tvecs = best_fit_region_center_for_v6(
        obj_model_region_center, 
        observed_pts,
        obj_model_region,
        )
    success = True
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success


def best_fit_region_center_for_v6(
        obj_model_region_center,
        observed_pts,
        obj_model_region,
        ):
    R = np.zeros((3,3))
    t = np.zeros((3,1))

    keep = np.ones(len(observed_pts), dtype=bool)
    keeps = []
    for i, (region_center, regions) in enumerate(zip(obj_model_region_center, obj_model_region)):
        region_num = max(pow(2, 5-i), 1)
        region_center = np.stack(region_center)
        regions = np.stack(regions)  # [npts, n_region_points, 3]
        R, t = best_fit_transform(region_center[keep], observed_pts[keep])
        transformed_region = R[None, None] @ regions[:, :region_num, :, None] + t.reshape(1, 1, 3, 1)
        dist = np.linalg.norm(transformed_region[:,:,:,0] - observed_pts[:, None, :], axis=-1)
        min_dist = np.min(dist, axis=1)
        keep = min_dist < np.median(min_dist)
        keeps.append(keep.copy())
    return R, t


def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7(
        input_cloud,
        mask_probability,
        code_probability,
        bit2class_id_center_and_region,
        dict_class_id_3D_points,
        region_bit,
        threshold,
        mean,
        uncertain_threshold,
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None,
        t_gt=None,
        rgb=None,
        depth=None,
        K=None,
        debug_image_dir=None,
        renderer=None,
        ):
    """
    
    """
    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    # input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]
    keep_i = []
    for i, c in enumerate(class_id):
        if not np.any(np.isnan(dict_class_id_3D_points[c][0])):
            keep_i.append(i)
    keep_i = np.array(keep_i)
    observed_pts = observed_pts[keep_i]
    class_id = class_id[keep_i]
    code_probability = code_probability[keep_i]
    mask_probability = mask_probability[keep_i]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    # prepare corresponding region and center for different correspondence, set first n bits whose number is not in [0.48, 0.52] as init region.
    iter_num = 16 + 1 - region_bit
    obj_model_region_center = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [3,]
    obj_model_region = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [varied_, 3]
    obj_model_region_bit_index = []  # a list whose num is n, the last index in bits which is outside [0.48, 0.52]

    for c_id, prob in zip(class_id, code_probability):
        index = region_bit
        for j in range(region_bit, 16):
            if 0.5-uncertain_threshold < prob[j] < 0.5+uncertain_threshold:
                break
            else:
                index = j
        obj_model_region_bit_index.append(index)
        for iter in range(0, iter_num):
            region_class_id = c_id >> max(15 - index - iter, 0)
            bit = 16 - max(15 - index - iter, 0)
            region = bit2class_id_center_and_region[bit][region_class_id]['region']
            obj_model_region[iter].append(region)
            center = bit2class_id_center_and_region[bit][region_class_id]['center']
            obj_model_region_center[iter].append(center)

    rot, tvecs = best_fit_region_center_for_v7_filter_in_end_new(
        obj_model_region_center, 
        observed_pts,
        obj_model_region,
        obj_model_region_bit_index,
        region_bit=region_bit,
        threshold=threshold,
        mean=mean,
        )
    
    success = True
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success


def best_fit_region_center_for_v7(
        obj_model_region_center,
        observed_pts,
        obj_model_region,
        obj_model_region_bit_index,
        ):
    print("warning: this is not the latest code")
    R = np.zeros((3,3))
    t = np.zeros((3,1))
    obj_model_region_bit_index = np.array(obj_model_region_bit_index)
    keep = np.ones(len(observed_pts), dtype=bool)
    keeps = []
    for i, (region_center, regions) in enumerate(zip(obj_model_region_center, obj_model_region)):
        region_num = max(pow(2, 5-i), 1)
        bit_index = min(i + 10, 16)  
        use_for_count_inlier = obj_model_region_bit_index > bit_index
        region_center = np.stack(region_center)
        regions = np.stack(regions)  # [npts, n_region_points, 3]
        R, t = best_fit_transform(region_center[keep], observed_pts[keep])
        transformed_region = R[None, None] @ regions[:, :region_num, :, None] + t.reshape(1, 1, 3, 1)
        dist = np.linalg.norm(transformed_region[:,:,:,0] - observed_pts[:, None, :], axis=-1)
        min_dist = np.min(dist, axis=1)
        keep = np.logical_or(min_dist < np.median(min_dist), use_for_count_inlier)
        keeps.append(keep.copy())
    return R, t


def best_fit_region_center_for_v7_filter_in_end_new(
        obj_model_region_center,
        observed_pts,
        obj_model_region,
        obj_model_region_bit_index,
        region_bit,
        threshold,
        mean,
        score = None,
        ):
    R = np.zeros((3,3))
    t = np.zeros((3,1))
    obj_model_region_bit_index = np.array(obj_model_region_bit_index)
    keep = np.ones(len(observed_pts), dtype=bool)

    keeps = []
    for i, (region_center, regions) in enumerate(zip(obj_model_region_center, obj_model_region)):
        region_num = max(pow(2, 15-region_bit-i), 1)
        bit_index = min(i + region_bit, 16)  
        use_for_count_inlier = obj_model_region_bit_index > bit_index
        region_center = np.stack(region_center)
        regions = np.stack(regions)  # [npts, n_region_points, 3]
        R, t = best_fit_transform(region_center[keep], observed_pts[keep])
        transformed_region = R[None, None] @ regions[:, :region_num, :, None] + t.reshape(1, 1, 3, 1)
        dist = np.linalg.norm(transformed_region[:,:,:,0] - observed_pts[:, None, :], axis=-1)
        min_dist = np.min(dist, axis=1)
        if mean:
            keep = min_dist < np.mean(min_dist)
        else:
            keep = min_dist < np.percentile(min_dist, threshold)

        if bit_index!=16:
            keep_for_filter = np.logical_or(keep, use_for_count_inlier)
        else:
            keep_for_filter = keep
        keeps.append(keep_for_filter.copy())

    keep_ = np.ones(len(observed_pts), dtype=bool)
    for i in range(len(keeps)):
        keep_ = np.logical_and(keep_, keeps[i])
    
    if keep_.sum() > 4:
        R, t = best_fit_transform(region_center[keep_], observed_pts[keep_])
    return R, t

def CNN_outputs_to_object_pose_with_uncertainty_hierarchy_v7_plus(
        input_cloud,
        mask_probability,
        code_probability,
        bit2class_id_center_and_region,
        dict_class_id_3D_points,
        region_bit,
        threshold,
        mean,
        class_base = 2,
        mask_target=None,  # ground truth mask
        class_code_image_target=None,   # ground truth code
        R_gt=None,
        t_gt=None,
        rgb=None,
        depth=None,
        K=None,
        debug_image_dir=None,
        renderer=None,
        ):
    """
    """
    # handling if mask_probability.shape == [1, npts]
    if mask_probability.shape[0] == 1:
        mask_probability = mask_probability.transpose(1, 0)  # [npts, 1]
    # handling if code_probability.shape == [16, npts]
    if code_probability.shape[0] == 16:
        code_probability = code_probability.transpose(1, 0)  # [npts, 16]

    obj_pts_idx = mask_probability[:, 0] > 0.5
    input_cloud = input_cloud * 1000.
    observed_pts = input_cloud[obj_pts_idx]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    # input_cloud = input_cloud[~ obj_pts_idx]
    mask_probability = mask_probability[obj_pts_idx]
    code_probability = code_probability[obj_pts_idx]
    code_image = np.zeros_like(code_probability, dtype=np.uint8)
    code_image[code_probability > 0.5] = 1
    class_id = class_code_images_to_class_id_image(code_image[None], class_base)[0]
    keep_i = []
    for i, c in enumerate(class_id):
        if not np.any(np.isnan(dict_class_id_3D_points[c][0])):
            keep_i.append(i)
    keep_i = np.array(keep_i)
    observed_pts = observed_pts[keep_i]
    class_id = class_id[keep_i]
    code_probability = code_probability[keep_i]
    mask_probability = mask_probability[keep_i]
    num = len(observed_pts)
    if num <= 4:
        return None, None, False

    # prepare corresponding region and center for different correspondence, set first n bits whose number is not in [0.48, 0.52] as init region.
    iter_num = 16 + 1 - region_bit
    obj_model_region_center = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [3,]
    obj_model_region = [[] for i in range(iter_num)]  # a list whose num is 7, each item contains n array of shape [varied_, 3]
    obj_model_region_bit_index = []  # a list whose num is n, the last index in bits which is outside [0.48, 0.52]

    for c_id, prob in zip(class_id, code_probability):
        index = region_bit
        for j in range(region_bit, 16):
            if 0.48 < prob[j] < 0.52:
                break
            else:
                index = j
        obj_model_region_bit_index.append(index)
        for iter in range(0, iter_num):
            region_class_id = c_id >> max(15 - index - iter, 0)
            bit = 16 - max(15 - index - iter, 0)
            region = bit2class_id_center_and_region[bit][region_class_id]['region']
            obj_model_region[iter].append(region)
            center = bit2class_id_center_and_region[bit][region_class_id]['center']
            obj_model_region_center[iter].append(center)

    rot, tvecs = best_fit_region_center_for_v7_filter_in_end_new_plus(
        obj_model_region_center, 
        observed_pts,
        obj_model_region,
        obj_model_region_bit_index,
        region_bit=region_bit,
        threshold=threshold,
        mean=mean,
        )
    success = True
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success

def best_fit_region_center_for_v7_filter_in_end_new_plus(
        obj_model_region_center,
        observed_pts,
        obj_model_region,
        obj_model_region_bit_index,
        region_bit,
        threshold,
        mean,
        score = None,
        ):
    R = np.zeros((3,3))
    t = np.zeros((3,1))
    obj_model_region_bit_index = np.array(obj_model_region_bit_index)
    keep = np.ones(len(observed_pts), dtype=bool)

    keeps = []
    for i, (region_center, regions) in enumerate(zip(obj_model_region_center, obj_model_region)):
        region_num = max(pow(2, 15-region_bit-i), 1)
        bit_index = min(i + region_bit, 16)  
        use_for_count_inlier = obj_model_region_bit_index > bit_index
        region_center = np.stack(region_center)
        regions = np.stack(regions)  # [npts, n_region_points, 3]
        R, t = best_fit_transform(region_center[keep], observed_pts[keep])
        transformed_region = R[None, None] @ regions[:, :region_num, :, None] + t.reshape(1, 1, 3, 1)
        dist = np.linalg.norm(transformed_region[:,:,:,0] - observed_pts[:, None, :], axis=-1)
        min_dist = np.min(dist, axis=1)
        max_dist = np.max(dist, axis=1)
        min_dist = (min_dist + max_dist) / 2
        if mean:
            keep = min_dist < np.mean(min_dist)
        else:
            keep = min_dist < np.percentile(min_dist, threshold)

        if bit_index!=16:
            keep_for_filter = np.logical_or(keep, use_for_count_inlier)
        else:
            keep_for_filter = keep
        keeps.append(keep_for_filter.copy())

    keep_ = np.ones(len(observed_pts), dtype=bool)
    for i in range(len(keeps)):
        keep_ = np.logical_and(keep_, keeps[i])
    
    if keep_.sum() > 4:
        R, t = best_fit_transform(region_center[keep_], observed_pts[keep_])
    return R, t