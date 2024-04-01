import torch
import numpy as np

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.CNN_output_to_pose import CNN_outputs_to_object_pose

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from tqdm import tqdm

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


def test_network_with_single_obj(
        net, dataloader, obj_diameter, writer, dict_class_id_3D_points, vertices, step, configs, ignore_n_bit=0,calc_add_and_adi=False):
    
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']
    binary_code_length = configs['binary_code_length']
    divide_number_each_itration = int(configs['divide_number_each_itration'])
    obj_name = configs['obj_name']
    dataset_name=configs['dataset_name']
    shift_center =configs['shift_center']
    _, symmetry_obj = get_obj_info(dataset_name)
    
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    #net to eval model
    net.eval()

    #test with test data
    ADX_passed=np.zeros(len(dataloader.dataset))
    ADX_error=np.zeros(len(dataloader.dataset))
    if calc_add_and_adi:
        ADY_passed=np.zeros(len(dataloader.dataset))
        ADY_error=np.zeros(len(dataloader.dataset))

    print("test dataset", flush=True)
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        # do the prediction and get the predicted binary code
        if not inputs: # no valid detected bbox
            continue
        if torch.cuda.is_available():
            for key in inputs:
                    inputs[key] = inputs[key].cuda()
    
        pred_masks_prob, pred_code_prob = net(inputs)
        pred_masks_prob = torch.torch.unsqueeze(pred_masks_prob, dim=-1)
        pred_code_prob = torch.torch.unsqueeze(pred_code_prob, dim=-1)

        pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_itration, binary_code_length=binary_code_length)
        pred_masks = from_output_to_class_mask(pred_masks_prob)

        # from binary code to pose
        pred_codes = pred_codes.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1)

        targets['Rs'] = targets['Rs'].detach().cpu().numpy()
        targets['ts'] = targets['ts'].detach().cpu().numpy()
        for counter, (r_GT, t_GT) in enumerate(zip(targets['Rs'] , targets['ts'] )):
            R_predict, t_predict, success = CNN_outputs_to_object_pose(inputs['cld_xyz0'].detach().cpu().numpy()[counter],pred_masks[counter], pred_codes[counter], dict_class_id_3D_points=dict_class_id_3D_points) 
            if shift_center:
                t_predict = t_predict + 1000. * inputs['original_center'].detach().cpu().numpy()[counter]

            batchsize = dataloader.batch_size
            sample_idx = batch_idx * batchsize + counter
            
            adx_error = 10000
            if success:
                adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(adx_error):
                    adx_error = 10000
            if adx_error < obj_diameter*0.1:
                ADX_passed[sample_idx] = 1
            ADX_error[sample_idx] = adx_error
           
            if calc_add_and_adi:
                ady_error = 10000
                if success:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*0.1:
                    ADY_passed[sample_idx] = 1
                ADY_error[sample_idx] = ady_error
    
    AUC_ADX_score = compute_auc_posecnn(ADX_error/1000.)
    ADX_passed = np.mean(ADX_passed)
    ADX_error= np.mean(ADX_error)

    writer.add_scalar('TESTDATA_{}/{}_test'.format(main_metric_name,main_metric_name), ADX_passed, step)
    writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(main_metric_name,main_metric_name), ADX_error, step)
    writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(main_metric_name,main_metric_name), AUC_ADX_score, step)
    if calc_add_and_adi:
        AUC_ADY_score = compute_auc_posecnn(ADY_error/1000.)
        ADY_passed = np.mean(ADY_passed)
        ADY_error= np.mean(ADY_error)
        writer.add_scalar('TESTDATA_{}/{}_test'.format(supp_metric_name,supp_metric_name), ADY_passed, step)
        writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(supp_metric_name,supp_metric_name), ADY_error, step)
        writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(supp_metric_name,supp_metric_name), AUC_ADY_score, step)

    #net back to train mode
    net.train()
    
    return ADX_passed, AUC_ADX_score