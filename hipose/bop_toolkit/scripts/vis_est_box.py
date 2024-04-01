# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in pose estimates saved in the BOP format."""

import os
import numpy as np
import itertools
import cv2
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visualization
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

# PARAMETERS.
################################################################################
p = {
  # Top N pose estimates (with the highest score) to be visualized for each
  # object in each image.
  'n_top': 1,  # 0 = all estimates, -1 = given by the number of GT poses.

  # True = one visualization for each (im_id, obj_id), False = one per im_id.
  'vis_per_obj_id': False,

  # Indicates whether to render RGB image.
  'vis_rgb': True,

  # Indicates whether to resolve visibility in the rendered RGB images (using
  # depth renderings). If True, only the part of object surface, which is not
  # occluded by any other modeled object, is visible. If False, RGB renderings
  # of individual objects are blended together.
  'vis_rgb_resolve_visib': True,

  # Indicates whether to render depth image.
  'vis_depth_diff': False,

  # If to use the original model color.
  'vis_orig_color': False,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  # Names of files with pose estimates to visualize (assumed to be stored in
  # folder config.eval_path). See docs/bop_challenge_2019.md for a description
  # of the format. Example results can be found at:
  # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
  'result_filenames': [
    '/home/z3d/zebrapose/detection_results/gdrnpp/yolox_lmo_pbr_self_defined.json',
  ],

  # Folder containing the BOP datasets.
  'datasets_path': r'/home/z3d/data',

  # Folder for output visualisations.
  'vis_path': os.path.join('/home/z3d/data/6dof_pose_experiments/experiments/self_defined/pose_result_bop', 'vis_est_box'),

  # Path templates for output images.
  'vis_rgb_tpath': os.path.join(
    '{vis_path}', '{result_name}', '{scene_id:06d}', '{vis_name}.jpg'),
  'vis_depth_diff_tpath': os.path.join(
    '{vis_path}', '{result_name}', '{scene_id:06d}',
    '{vis_name}_depth_diff.jpg'),
}
################################################################################


# Load colors.
colors_path = os.path.join(
  os.path.dirname(visualization.__file__), 'colors.json')
colors = inout.load_json(colors_path)

for result_fname in p['result_filenames']:
  misc.log('Processing: ' + result_fname)

  # Parse info about the method and the dataset from the filename.
  result_name = os.path.splitext(os.path.basename(result_fname))[0]
  
  result_info = result_name.split('_')
  method = result_info[0]

  dataset = result_info[1]
  split = "test"
  split_type = None

  # Load dataset parameters.
  dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

  model_type = 'vis'
  dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

  # Rendering mode.
  renderer_modalities = []
  if p['vis_rgb']:
    renderer_modalities.append('rgb')
  if p['vis_depth_diff'] or (p['vis_rgb'] and p['vis_rgb_resolve_visib']):
    renderer_modalities.append('depth')
  renderer_mode = '+'.join(renderer_modalities)

  # Create a renderer.
  width, height = dp_split['im_size']
  width = 1920
  height = 1080

  # Load pose estimates.
  misc.log('Loading pose estimates...')
  ests = inout.load_json(os.path.join(config.results_path, result_fname))

  # Organize the pose estimates by scene, image and object.
  misc.log('Organizing result...')
  ests_org = {}
  for est in ests:
    ests_org.setdefault(est['scene_id'], {}).setdefault(
      est['image_id'], {}).setdefault(est['category_id'], []).append(est)

  for scene_id, scene_ests in ests_org.items():
    for im_ind, (im_id, im_ests) in enumerate(scene_ests.items()):

      if im_ind % 10 == 0:
        split_type_str = ' - ' + split_type if split_type is not None else ''
        misc.log(
          'Visualizing pose estimates - method: {}, dataset: {}{}, scene: {}, '
          'im: {}'.format(method, dataset, split_type_str, scene_id, im_id))
      # Intrinsic camera matrix.

      im_ests_vis = []
      im_ests_vis_obj_ids = []
      for obj_id, obj_ests in im_ests.items():

        # Sort the estimates by score (in descending order).
        obj_ests_sorted = sorted(
          obj_ests, key=lambda est: est['score'], reverse=True)

        # Select the number of top estimated poses to visualize.
        if p['n_top'] == 0:  # All estimates are considered.
          n_top_curr = None
        else:  # Specified by the parameter n_top.
          n_top_curr = p['n_top']
        obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]

        # Get list of poses to visualize.
        for est in obj_ests_sorted:
          est['obj_id'] = obj_id

          # Text info to write on the image at the pose estimate.
          if p['vis_per_obj_id']:
            est['text_info'] = [
              {'name': '', 'val': est['score'], 'fmt': ':.2f'}
            ]
          else:
            val = '{}:{:.2f}'.format(obj_id, est['score'])
            est['text_info'] = [{'name': '', 'val': val, 'fmt': ''}]

        im_ests_vis.append(obj_ests_sorted)
        im_ests_vis_obj_ids.append(obj_id)

      # Join the per-object estimates if only one visualization is to be made.
      if not p['vis_per_obj_id']:
        im_ests_vis = [list(itertools.chain.from_iterable(im_ests_vis))]

      for ests_vis_id, ests_vis in enumerate(im_ests_vis):

        # Load the color and depth images and prepare images for rendering.
        rgb = None
        if p['vis_rgb']:
          if 'rgb' in dp_split['im_modalities']:
            rgb = inout.load_im(dp_split['rgb_tpath'].format(
              scene_id=scene_id, im_id=im_id))[:, :, :3]
          elif 'gray' in dp_split['im_modalities']:
            gray = inout.load_im(dp_split['gray_tpath'].format(
              scene_id=scene_id, im_id=im_id))
            rgb = np.dstack([gray, gray, gray])
          else:
            raise ValueError('RGB nor gray images are available.')


        # Visualization name.
        if p['vis_per_obj_id']:
          vis_name = '{im_id:06d}_{obj_id:06d}'.format(
            im_id=im_id, obj_id=im_ests_vis_obj_ids[ests_vis_id])
        else:
          vis_name = '{im_id:06d}'.format(im_id=im_id)

        # Path to the output RGB visualization.
        vis_rgb_path = None
        if p['vis_rgb']:
          vis_rgb_path = p['vis_rgb_tpath'].format(
            vis_path=p['vis_path'], result_name=result_name, scene_id=scene_id,
            vis_name=vis_name)
        
        
        # Visualization.
        im_size = (rgb.shape[1], rgb.shape[0])
        ren_rgb = np.zeros(rgb.shape, np.uint8)
        ren_rgb_info = np.zeros(rgb.shape, np.uint8)
        for est in ests_vis:
          cls_id = est['category_id']
          if cls_id !=6:
            continue
          box = est['bbox']
          x0 = int(box[0])
          y0 = int(box[1])
          x1 = int(box[2]) + x0
          y1 = int(box[3]) + y0
          color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
          text = "{}:{:.1f}%".format(cls_id, est['score'] * 100)
          txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
          font = cv2.FONT_HERSHEY_SIMPLEX
          txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
          cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
          txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
          cv2.rectangle(
            rgb,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
          cv2.putText(rgb, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
          misc.ensure_dir(os.path.dirname(vis_rgb_path))
          inout.save_im(vis_rgb_path, rgb.astype(np.uint8), jpg_quality=95)
misc.log('Done.')
