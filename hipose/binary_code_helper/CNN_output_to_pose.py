from binary_code_helper.class_id_encoder_decoder import class_code_images_to_class_id_image
import numpy as np
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


def load_dict_class_id_3D_points_withNorm(path):
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
            code, x, y, z, nx, ny, nz= line.split(" ")
            code = float(code)
            x = float(x)
            y = float(y)
            z = float(z)
            nx = float(nx)
            ny = float(ny)
            nz = float(nz)

            dict_class_id_3D_points[code] = np.array([x,y,z, nx, ny, nz])

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

def CNN_outputs_to_object_pose(input_cloud, mask_image, class_code_image, class_base=2, dict_class_id_3D_points=None, class_code_image_target=None, R_gt=None, t_gt=None):
    obj_pts_idx = mask_image.nonzero()[0]
    class_code_image = class_code_image[obj_pts_idx]
    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)
    class_id_image = class_id_image.squeeze(-1)
    mask_image = mask_image.squeeze(-1)

    observed_obj_pts = input_cloud[obj_pts_idx] * 1000. # in mm

    obj_model_pts = np.zeros((class_id_image.shape[0], 3))

    success=True
    for i, id in enumerate(class_id_image):
        obj_model_pts[i] = dict_class_id_3D_points[id]

    if True:
        rot, tvecs = baseline_solver(obj_model_pts, observed_obj_pts, inlier_threshold=3, num_sub_sample_pts=20)
    else:
        rot, tvecs = RANSAC_best_fit_v2(obj_model_pts, observed_obj_pts, inlier_threshold=3, num_sub_sample_pts=20)
    if np.isnan(tvecs).any():
        success = False
    visualize = False
    if visualize:
        obj_model_pts = np.matmul(obj_model_pts, rot.T) + tvecs
        vis_input_cloud(input_cloud, observed_obj_pts, obj_model_pts)
    visualize_for_error = False
    if visualize_for_error:
        print("visualize for error")
        class_code_image_target = class_code_image_target[obj_pts_idx]
        class_id_image_target = class_code_images_to_class_id_image(class_code_image_target, class_base)
        class_id_image_target = class_id_image_target.squeeze(-1)

        obj_model_pts_target = np.zeros((class_id_image_target.shape[0], 3))
        success=True
        for i, id in enumerate(class_id_image_target):
            obj_model_pts_target[i] = dict_class_id_3D_points[id]
        obj_model_pts_diff_bit = np.bitwise_xor(class_code_image_target.astype(np.uint8), class_code_image.astype(np.uint8))
        obj_model_pts_error_bit = np.ones(obj_model_pts_diff_bit.shape[0], dtype=np.uint8) * 16
        for i in range(obj_model_pts_diff_bit.shape[0]):
            for j in range(16):
                if obj_model_pts_diff_bit[i, 0, j] == 1:
                    obj_model_pts_error_bit[i] = j
                    break
        obj_model_pts = np.matmul(obj_model_pts, rot.T) + tvecs
        gt_model_pts = np.matmul(obj_model_pts_target, R_gt.T) + t_gt[:, 0]
        vis_error(observed_obj_pts, obj_model_pts, gt_model_pts, obj_model_pts_error_bit)

    return rot, tvecs.reshape((3,1)), success


def RANSAC_best_fit(A, B, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30):
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
                best_rot = R
                best_tvecs = t
    
    else:
        try:
            best_rot, best_tvecs = best_fit_transform(A, B)
        except:
            return best_rot, best_tvecs
    #print(max_number_inliers)
    return best_rot, best_tvecs


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


def RANSAC_best_fit_v3(A, B, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30, min_inlier = 0.5):
    max_number_inliers = 0
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    max_inlier_rot = np.zeros((3,3))
    max_inlier_tvecs = np.zeros((3,1))

    min_dist = np.finfo(np.float64).max
    # print(A.shape)  N x 3
    lens_samples = len(A)
    min_number_inlier = min_inlier * lens_samples

    success = False

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
         

            if n_inlier >= min_number_inlier:
                # solve R and t with all inliers
                inlier_A = A[dist < inlier_threshold]  
                inlier_B = B[dist < inlier_threshold]  
                try:
                    R, t = best_fit_transform(inlier_A, inlier_B)
                    transformed_A = R @ A.transpose()  + t.reshape(3,1)
                    dist = np.linalg.norm(transformed_A.transpose() - B, axis=1) 
                    inlier_mean_dist = np.mean(dist[dist < inlier_threshold])
                    if inlier_mean_dist < min_dist:
                        min_dist = inlier_mean_dist
                        best_rot = R
                        best_tvecs = t
                        success = True
                except:    
                    best_rot = R
                    best_tvecs = t

            #if n_inlier > max_number_inliers:
            #    max_number_inliers = n_inlier
            #    max_inlier_rot = R
            #    max_inlier_tvecs = t     

        #if success == False:
        #    best_rot = max_inlier_rot
        #    best_tvecs = max_inlier_tvecs

    else:
        try:
            best_rot, best_tvecs = best_fit_transform(A, B)
        except:
            return best_rot, best_tvecs
    #print(max_number_inliers)
    return best_rot, best_tvecs

def best_fit_transform_withNorm(dest_points, source_points, R_dest2source, t_dest2source):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points dest_points to source_points in m spatial dimensions
    Input:
        dest_points: Nx6 numpy array of corresponding points, with normals, usually points on mdl
        source_points: Nx3 numpy array of corresponding points, usually points on camera axis
    Returns:
    R_init: 3x3 rotation matrix
    t_init: 3x1 translation vector
    '''
    assert dest_points.shape[0] == source_points.shape[0]
    num_points = dest_points.shape[0]
    if num_points < 4:
        return R_dest2source, t_dest2source

    R = R_dest2source.transpose()
    t = -np.matmul(R, t_dest2source)

    best_R = R.copy()
    best_t = t.copy()

    best_cost = 1e9

    for it in range(10):
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        cost = 0
        for i in range(num_points):
            P_ = np.matmul(R, source_points[[i], :].transpose()) + t
            e = np.matmul(dest_points[[i], 3:], P_-dest_points[[i], :3].transpose())
            cost += np.linalg.norm(e)

            x, y, z  = source_points[i, 0], source_points[i, 1], source_points[i, 2]
            de_dtheta = dest_points[[i], 3:] @ R @ np.column_stack([np.array([[0, z, -y], [-z, 0, x], [y, -x, 0]]), np.identity(3)])

            b -= np.matmul(de_dtheta.transpose(), e)
            H += np.matmul(de_dtheta.transpose(), de_dtheta)
        cost = cost / num_points
        if it == 0:
            best_cost = cost

        debug = True
        if debug:
            if it == 0:
                print("it", 0, "init_cost", best_cost)
            else:
                print("it", it, "cost", cost)

        if cost < best_cost:
            best_cost = cost
            best_R = R.copy()
            best_t = t.copy()

        #  Optimize and update pose
        try:
            theta = np.linalg.solve(H, b)
        except:
            return R_dest2source, t_dest2source
        dR, _ = cv2.Rodrigues(theta[0:3])
        dt = theta[3:6]
            
        t = np.matmul(R, dt) + t
        R = np.matmul(R, dR)
    cost = 0
    for i in range(num_points):
        P_ = np.matmul(R, dest_points[[i], :3].transpose()) + t
        e = np.matmul(dest_points[[i], 3:], P_-source_points[[i], :].transpose())
        cost += np.linalg.norm(e)
    cost = cost / num_points
    if cost < best_cost:
        best_cost = cost
        best_R = R.copy()
        best_t = t.copy()

    R_dest2source = best_R.transpose()
    t_dest2source = -R_dest2source @ best_t
    return R_dest2source, t_dest2source

def RANSAC_best_fit_withNorm_v1(A, B, best_rot, best_tvecs, max_num_itrations=1, inlier_threshold=3, num_sub_sample_pts=30):
    transformed_A = best_rot @ A[:, :3].transpose()  + best_tvecs.reshape(3,1)
    dist = np.linalg.norm(transformed_A.transpose() - B, axis=1)
    inlier_A = A[dist < 11]  
    inlier_B = B[dist < 11]  
    R, t = best_fit_transform_withNorm(inlier_A, inlier_B, best_rot, best_tvecs)
    return R, t


def CNN_outputs_to_object_pose_v2(input_cloud, mask_image, class_code_image, R_init, t_init, class_base=2, dict_class_id_3D_points_v2=None):
    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)
    class_id_image = class_id_image.squeeze(-1)
    mask_image = mask_image.squeeze(-1)

    obj_pts_idx = mask_image.nonzero()

    observed_obj_pts = input_cloud[obj_pts_idx] * 1000. # in mm
    class_id_image = class_id_image[obj_pts_idx]

    obj_model_pts = np.zeros((class_id_image.shape[0], 6))
    success=True
    for i, id in enumerate(class_id_image):
        obj_model_pts[i] = dict_class_id_3D_points_v2[id]

    rot, tvecs = RANSAC_best_fit_withNorm_v1(obj_model_pts, observed_obj_pts, R_init, t_init, inlier_threshold=3, num_sub_sample_pts=20)  
    visualize = False
    if visualize:
        print("v2")
        obj_model_pts = np.matmul(obj_model_pts[:, :3], rot.T) + tvecs.reshape(1, 3)
        vis_input_cloud(input_cloud, observed_obj_pts, obj_model_pts[:, :3])
    return rot, tvecs.reshape((3,1)), success


def CNN_outputs_to_object_pose_v3(input_cloud, mask_image, class_code_image, R_init, t_init, bit=10, class_base=2, dict_class_id_3D_points_v2=None):
    obj_pts_idx = mask_image.nonzero()[0]

    class_code_image = class_code_image[obj_pts_idx]  # shape [N,1,16]
    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)  # shape [N,1]
    class_id_image = class_id_image.squeeze(-1)  # shape [N]

    class_code_floor = np.zeros_like(class_code_image, dtype=np.uint32)
    class_code_floor[..., :bit] = class_code_image[..., :bit]
    class_id_floor = class_code_images_to_class_id_image(class_code_floor, class_base)
    class_id_floor = class_id_floor.squeeze(-1)

    class_code_ceil = np.ones_like(class_code_image, dtype=np.uint32)
    class_code_ceil[..., :bit] = class_code_image[..., :bit]
    class_id_ceil = class_code_images_to_class_id_image(class_code_ceil, class_base)
    class_id_ceil = class_id_ceil.squeeze(-1)  # shape [N]

    observed_obj_pts = input_cloud[obj_pts_idx] * 1000. # in mm

    # target points in object model
    obj_model_pts = np.zeros((class_id_image.shape[0], 6))
    # target nearest points in region
    obj_region_pts = []  # shape = (num_point, num_region, 6)

    ## TODO, need to find the most nearest points
    success=True
    for i, id in enumerate(class_id_image):
        id_region_pts = []
        obj_model_pts[i] = dict_class_id_3D_points_v2[id]
        for region_id in range(class_id_floor[i], class_id_ceil[i]+1):
            id_region_pts.append(dict_class_id_3D_points_v2[region_id])
        obj_region_pts.append(id_region_pts)
    obj_region_pts = np.asarray(obj_region_pts)

    transformed_obj_region_pts = R_init[None, None, ...] @ obj_region_pts[..., :3, None]  + t_init.reshape(3,1)[None, None, ...]
    dist = np.linalg.norm(transformed_obj_region_pts[:,:,:,0] - observed_obj_pts[:, None, :], axis=-1)
    idx = np.argmin(dist, axis=1)
    min_dist = np.min(dist, axis=1)
    nearest_points = obj_region_pts[np.arange(len(obj_region_pts)),idx]
    distance_between_nearest_points_and_obj_model_pts = np.linalg.norm(nearest_points[:, :3] - obj_model_pts[:,:3], axis=1)
    rot, tvecs = RANSAC_best_fit_v2(nearest_points[:,:3], observed_obj_pts, num_sub_sample_pts=30)  
    visualize = False
    if visualize:
        print("v2")
        obj_model_pts = np.matmul(obj_model_pts[:, :3], rot.T) + tvecs.reshape(1, 3)
        vis_input_cloud(input_cloud, observed_obj_pts, obj_model_pts[:, :3])
    return rot, tvecs.reshape((3,1)), success


def CNN_outputs_to_object_pose_with_uncertainty_v1(input_cloud, mask_image, mask_score, class_code_image, class_code_score, class_base=2, dict_class_id_3D_points=None, class_code_image_target=None, R_gt=None, t_gt=None, score_bit=0):
    def calculate_score(mask_score, class_code_score, score_bit=0):
        score = np.copy(mask_score)
        for i in range(score_bit):
            score = score * class_code_score[:, :, i] * 0.5^(i+1)
        score = score / np.sum(score)
        score = score.squeeze(-1)
        return score
    obj_pts_idx = mask_image.nonzero()[0]
    class_code_image = class_code_image[obj_pts_idx]
    mask_score = mask_score[obj_pts_idx]
    class_code_score = class_code_score[obj_pts_idx]
    class_id_image = class_code_images_to_class_id_image(class_code_image, class_base)
    class_id_image = class_id_image.squeeze(-1)
    # calculate score
    score = calculate_score(mask_score, class_code_score)
    observed_obj_pts = input_cloud[obj_pts_idx] * 1000. # in mm

    obj_model_pts = np.zeros((class_id_image.shape[0], 3))

    success=True
    for i, id in enumerate(class_id_image):
        obj_model_pts[i] = dict_class_id_3D_points[id]
    
    rot, tvecs = RANSAC_best_fit_v4(obj_model_pts, observed_obj_pts, score, inlier_threshold=3, num_sub_sample_pts=20)
    if np.isnan(tvecs).any():
        success = False
    return rot, tvecs.reshape((3,1)), success

def RANSAC_best_fit_v4(A, B, score, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30):
    max_number_inliers = 0
    best_rot = np.zeros((3,3))
    best_tvecs = np.zeros((3,1))

    # print(A.shape)  N x 3
    lens_samples = len(A)

    if lens_samples > num_sub_sample_pts:
        for i in range(max_num_itrations):
            index = np.random.choice(np.arange(lens_samples), num_sub_sample_pts, p=score)

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

    
def baseline_solver(source, target, max_num_itrations=300, inlier_threshold=3, num_sub_sample_pts=30):
    ###
    # hard code the numbers, those are better for this baseline solver
    max_num_itrations=2000 
    inlier_threshold=2
    num_sub_sample_pts=10
    ###

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    ransac_distance_threshold = inlier_threshold
    max_iteration = max_num_itrations
    ransac_n = num_sub_sample_pts
    known_correspondences = []

    for indx in range(len(source_pcd.points)):
        known_correspondences.append([indx,indx])
    
    correspondences = o3d.utility.Vector2iVector(known_correspondences)

    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(source_pcd, target_pcd, correspondences, ransac_distance_threshold, estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=ransac_n, criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=max_iteration, confidence=0.999) )
    final_transformation = ransac_result.transformation

    if final_transformation.shape[0] != 0:
        rot = final_transformation[0:3, :3]
        tvecs = final_transformation[0:3, 3]
        tvecs = tvecs.reshape((3,1))
        return rot, tvecs
    else:
        rot = np.zeros((3,3))
        tvecs = np.zeros((3,1))
        return rot, tvecs 