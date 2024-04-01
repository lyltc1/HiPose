import json

detection_results_file = "/home/z3d/zebrapose/detection_results/gdrnpp/yolox_lmo_pbr_self_defined.json"
assert "lmo" in detection_results_file
contineous_id_to_catogory_id = {1:1,2:5,3:6,4:8, 5:9, 6:10, 7:11, 8:12}

with open(detection_results_file) as jsonFile:
        detection_results = json.load(jsonFile)
        jsonFile.close()
for detection_result in detection_results:
    detection_result['category_id'] = contineous_id_to_catogory_id[detection_result['category_id']]

with open(detection_results_file, 'w') as f:
    json.dump(detection_results, f)