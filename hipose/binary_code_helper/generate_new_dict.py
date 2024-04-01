import numpy as np
from scipy.spatial.distance import cdist
from binary_code_helper.class_id_encoder_decoder import str_code_to_id

def generate_new_corres_dict_and_region_and_maxdist(full_binary_corres_dict, num_bit_old_dict, num_bit_new_dict):
    all_binary_codes = [[] for i in range(num_bit_old_dict)]
    all_binary_codes[0].append('0')
    all_binary_codes[0].append('1')
    
    for bit_number in range(1, num_bit_old_dict):
        for i in all_binary_codes[bit_number-1]:
            all_binary_codes[bit_number].append(i+'0')
            all_binary_codes[bit_number].append(i+'1')

    # find all related 16 bit binary code to each of the new code
    # a dict whose key is class_id, value is a list contains binary_id, like {0: [0, 1, ..., 63], 1:[64, ..., 127], ...}
    new_corres_dict = {}
    for index, new_binary_code in enumerate(all_binary_codes[num_bit_new_dict-1]):
        related_binary_codes = all_binary_codes[num_bit_old_dict-1][index*pow(2, (num_bit_old_dict-num_bit_new_dict)):(index+1)*pow(2, (num_bit_old_dict-num_bit_new_dict))]
        for rank, related_binary_code in enumerate(related_binary_codes):
            related_binary_id = str_code_to_id(related_binary_code)
            if rank == 0:
                new_corres_dict[str_code_to_id(new_binary_code)] = [related_binary_id]
            else:
                new_corres_dict[str_code_to_id(new_binary_code)].append(related_binary_id) 
    
    # caculate the center and region for class
    # like {0: {'center': np.array[1, 3], 'region': np.array[n, 3]},
    #       1: {'center': np.array[1, 3], 'region': np.array[n, 3]},
    #       ......
    #      }
    result = dict()
    for key, values in new_corres_dict.items():
        result[key] = {'center': np.zeros((1,3)), 
                       'region': np.zeros((len(values), 3)),
                       'maxdist': 100.}
        for i, value in enumerate(values):
            result[key]['region'][i] = full_binary_corres_dict[value]
        rows_with_nan = np.any(np.isnan(result[key]['region']), axis=1)
        result[key]['center'] = np.nanmean(result[key]['region'], axis=0)
        result[key]['region'][rows_with_nan] = result[key]['center']
        # Naive way of finding the best pair in O(H^2) time if H is number of points on hull
        hdist = cdist(result[key]['region'], result[key]['region'], metric='euclidean')
        result[key]['maxdist'] = np.max(hdist)
    return result

def generate_new_corres_dict_and_region(full_binary_corres_dict, num_bit_old_dict, num_bit_new_dict):
    all_binary_codes = [[] for i in range(num_bit_old_dict)]
    all_binary_codes[0].append('0')
    all_binary_codes[0].append('1')
    
    for bit_number in range(1, num_bit_old_dict):
        for i in all_binary_codes[bit_number-1]:
            all_binary_codes[bit_number].append(i+'0')
            all_binary_codes[bit_number].append(i+'1')

    # find all related 16 bit binary code to each of the new code
    # a dict whose key is class_id, value is a list contains binary_id, like {0: [0, 1, ..., 63], 1:[64, ..., 127], ...}
    new_corres_dict = {}
    for index, new_binary_code in enumerate(all_binary_codes[num_bit_new_dict-1]):
        related_binary_codes = all_binary_codes[num_bit_old_dict-1][index*pow(2, (num_bit_old_dict-num_bit_new_dict)):(index+1)*pow(2, (num_bit_old_dict-num_bit_new_dict))]
        for rank, related_binary_code in enumerate(related_binary_codes):
            related_binary_id = str_code_to_id(related_binary_code)
            if rank == 0:
                new_corres_dict[str_code_to_id(new_binary_code)] = [related_binary_id]
            else:
                new_corres_dict[str_code_to_id(new_binary_code)].append(related_binary_id) 
    
    # caculate the center and region for class
    # like {0: {'center': np.array[1, 3], 'region': np.array[n, 3]},
    #       1: {'center': np.array[1, 3], 'region': np.array[n, 3]},
    #       ......
    #      }
    result = dict()
    for key, values in new_corres_dict.items():
        result[key] = {'center': np.zeros((1,3)), 'region': np.zeros((len(values), 3))}
        for i, value in enumerate(values):
            result[key]['region'][i] = full_binary_corres_dict[value]
        rows_with_nan = np.any(np.isnan(result[key]['region']), axis=1)
        result[key]['center'] = np.nanmean(result[key]['region'], axis=0)
        result[key]['region'][rows_with_nan] = result[key]['center']
    return result


def generate_new_corres_dict(full_binary_corres_dict, num_bit_old_dict, num_bit_new_dict):
    all_binary_codes = [[] for i in range(num_bit_old_dict)]
    all_binary_codes[0].append('0')
    all_binary_codes[0].append('1')
    
    for bit_number in range(1, num_bit_old_dict):
        for i in all_binary_codes[bit_number-1]:
            all_binary_codes[bit_number].append(i+'0')
            all_binary_codes[bit_number].append(i+'1')

    # find all related 16 bit binary code to each of the new code
    new_corres_dict = {}
    for index, new_binary_code in enumerate(all_binary_codes[num_bit_new_dict-1]):
        related_binary_codes = all_binary_codes[num_bit_old_dict-1][index*pow(2, (num_bit_old_dict-num_bit_new_dict)):(index+1)*pow(2, (num_bit_old_dict-num_bit_new_dict))]
        for rank, related_binary_code in enumerate(related_binary_codes):
            related_binary_id = str_code_to_id(related_binary_code)
            if rank == 0:
                new_corres_dict[str_code_to_id(new_binary_code)] = [related_binary_id]
            else:
                new_corres_dict[str_code_to_id(new_binary_code)].append(related_binary_id) 
    
    # caculate the center of related correspoints
    for key, values in new_corres_dict.items():
        sum_corres_3D_points = np.zeros((1,3))
        for value in values:
            sum_corres_3D_points = sum_corres_3D_points + full_binary_corres_dict[value]
        mean_corres_3D_points = sum_corres_3D_points / len(values)
        new_corres_dict[key] = mean_corres_3D_points

    return new_corres_dict


