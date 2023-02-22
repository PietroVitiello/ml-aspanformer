import glob
import json

class nana():

    def __init__(self) -> None:
        pass

    def load_scene(self,
                scene_dir,
                load_intrinsic=False,
                load_image_resolution=False,
                load_T_WC=False,
                load_T_WO=False,
                load_T_CO=False,
                load_rgb=False,
                load_depth=False,
                load_target_obj_segmaps=False,
                load_instance_segmaps=False,
                load_instance_attribute_maps=False):
        data = {}

        if load_T_WO or load_T_CO or load_rgb or load_depth or load_target_obj_segmaps or load_instance_segmaps or load_instance_attribute_maps:
            n = len(glob.glob(os.path.join(scene_dir, 'colors', '*')))

        if load_intrinsic:
            data['intrinsic'] = np.load(os.path.join(scene_dir, 'intrinsic.npy'))
        if load_image_resolution:
            data['image_resolution'] = np.load(os.path.join(scene_dir, 'image_resolution.npy'))
        if load_T_WC:
            data['T_WC'] = np.load(os.path.join(scene_dir, 'T_WC_opencv.npy'))
        if load_T_WO:

            for i in range(n):
                with open(os.path.join(scene_dir, 'T_WO', f'{i}.json')) as jsonfile:
                    T_WO = json.load(jsonfile)
                for obj_name in T_WO:
                    T_WO[obj_name] = np.asarray(T_WO[obj_name])
                data[f'T_WO_{i}'] = T_WO

        if load_T_CO:
            for i in range(n):
                with open(os.path.join(scene_dir, 'T_CO', f'{i}.json')) as jsonfile:
                    T_WO = json.load(jsonfile)
                for obj_name in T_WO:
                    T_WO[obj_name] = np.asarray(T_WO[obj_name])
                data[f'T_CO_{i}'] = T_WO

        if load_rgb:
            for i in range(n):
                data[f'rgb_{i}'] = np.asarray(Image.open(os.path.join(scene_dir, 'colors', f'{i}' + '.png')))
        if load_depth:
            for i in range(n):
                data[f'depth_{i}'] = np.asarray(
                    Image.open(os.path.join(scene_dir, 'depth', f'{i}' + '.png'))).astype(np.uint16)
        if load_target_obj_segmaps:
            for i in range(n):
                data[f'target_obj_segmap_{i}'] = np.asarray(
                    Image.open(os.path.join(scene_dir, 'cp_main_obj_segmaps', f'{i}' + '.png'))).astype(bool)
        if load_instance_segmaps:
            for i in range(n):
                data[f'instance_segmap_{i}'] = np.asarray(
                    Image.open(os.path.join(scene_dir, 'instance_segmaps', f'{i}' + '.png')))
        if load_instance_attribute_maps:
            for i in range(n):
                with open(os.path.join(scene_dir, 'instance_attribute_maps', f'{i}.json')) as f:
                    data[f'instance_attribute_map_{i}'] = json.load(f)

        return data

    def __getattr__(self, idx):
        # Get all the data for current scene
        scene_dir = os.path.join(ASSETS_DIR, 'datasets', dataset_name, SCENE_SAVE_FORMAT.format(idx))
        data = self.load_scene(scene_dir, load_T_WC=True, load_rgb=True, load_depth=True, load_intrinsic=True,
                            load_T_CO=True, load_T_WO=True, load_target_obj_segmaps=True, load_instance_segmaps=True,
                            load_instance_attribute_maps=True)

        K = data['intrinsic'].copy()
        rgb_0 = data['rgb_0'].copy()
        rgb_1 = data['rgb_1'].copy()
        all_objs_T_CO_0 = data['T_CO_0'].copy()
        all_objs_T_CO_1 = data['T_CO_1'].copy()

        # # Preprocess VIT inputs
        # image0_batch, image0_pil = dino_extractor.preprocess_pil_image(Image.fromarray(rgb_0).convert('RGB'),
        #                                                                 load_size=load_size)
        # image1_batch, image1_pil = dino_extractor.preprocess_pil_image(Image.fromarray(rgb_1).convert('RGB'),
        #                                                                 load_size=load_size)

        # # Get desired input size, resize other images and adapt intrinsic matrix
        # _, _, inference_height, inference_width = image0_batch.shape

        # K = calculate_intrinsic_for_new_resolution(K, new_width=inference_width, new_height=inference_height,
        #                                             old_width=data['depth_0'].shape[1],
        #                                             old_height=data['depth_0'].shape[0])

        # rgb_0 = np.asarray(image0_pil)
        # rgb_1 = np.asarray(image1_pil)

        # start = time.time()

        # try:
        #     with torch.no_grad():
        #         points0, points1 = find_correspondences(extractor=dino_extractor,
        #                                                 image1_batch=image0_batch,
        #                                                 image2_batch=image1_batch,
        #                                                 num_pairs=num_pairs,
        #                                                 layer=layer,
        #                                                 facet=facet,
        #                                                 bin=bin,
        #                                                 thresh=thresh,
        #                                                 device=device)
        #     inference_time = time.time() - start
        # except Exception as e:
        #     print(e)
        #     continue

        # import matplotlib.pyplot as plt
        # fig0, fig1 = draw_correspondences(points0, points1, image0_pil, image1_pil)
        # plt.show()

        # points0 = np.asarray(points0)
        # points0 = np.concatenate((points0[:, 1:2], points0[:, 0:1]), axis=1)

        # points1 = np.asarray(points1)
        # points1 = np.concatenate((points1[:, 1:2], points1[:, 0:1]), axis=1)

        depth_0 = data['depth_0'].copy()
        depth_1 = data['depth_1'].copy()
        instance_segmap_0 = data['instance_segmap_0'].copy()
        instance_segmap_1 = data['instance_segmap_1'].copy()

        depth_0 = cv2.resize(depth_0, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        depth_1 = cv2.resize(depth_1, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        instance_segmap_0 = cv2.resize(instance_segmap_0, (inference_width, inference_height), 0, 0,
                                        interpolation=cv2.INTER_NEAREST)
        instance_segmap_1 = cv2.resize(instance_segmap_1, (inference_width, inference_height), 0, 0,
                                        interpolation=cv2.INTER_NEAREST)

        common_objs = get_common_objs(data['instance_attribute_map_0'], data['instance_attribute_map_1'])

        for obj in common_objs:
            name, idx = obj['name'], obj['idx']

            obj_segmap_0 = instance_segmap_0 == idx
            obj_segmap_1 = instance_segmap_1 == idx
            T_CO_0 = all_objs_T_CO_0[name]
            T_CO_1 = all_objs_T_CO_1[name]

            # Ensure that the object is fully visible
            if obj_segmap_0[:, 0].any() or \
                    obj_segmap_0[:, -1].any() or \
                    obj_segmap_0[0, :].any() or \
                    obj_segmap_0[-1, :].any() or \
                    obj_segmap_1[:, 0].any() or \
                    obj_segmap_1[:, -1].any() or \
                    obj_segmap_1[0, :].any() or \
                    obj_segmap_1[-1, :].any():
                continue

            T_delta = torch.tensor(T_CO_1 @ np.linalg.inv(T_CO_0), dtype=torch.float32).unsqueeze(0)

            # image = plot_matches(rgb_0, points0, rgb_1, points1, num_points_to_plot=-1)
            # plot_image(image, mode='TkAgg')

            obj_indices_0 = get_keypoint_indices(obj_segmap_0)
            obj_indices_1 = get_keypoint_indices(obj_segmap_1)

            valid_0 = np.zeros(len(points0), dtype=bool)

            for i, kpt in enumerate(points0):
                if (kpt == obj_indices_0).all(axis=1).any():
                    valid_0[i] = True

            valid_1 = np.zeros(len(points1), dtype=bool)

            for i, kpt in enumerate(points1):
                if (kpt == obj_indices_1).all(axis=1).any():
                    valid_1[i] = True

            valid = np.logical_and(valid_0, valid_1)

            if valid.sum() > 3:

                obj_points0 = points0[valid]
                obj_points1 = points1[valid]

                # Get error in the keypoint locations: TODO
                keypoints_1, valid_correspondence = estimate_correspondences(obj_points0,
                                                                                depth_0,
                                                                                depth_1,
                                                                                T_CO_0,
                                                                                T_CO_1,
                                                                                K,
                                                                                depth_rejection_threshold=0.02,
                                                                                return_valid_list=True,
                                                                                depth_units='mm')

                mean_pixel_error = np.abs(
                    obj_points1[valid_correspondence] - keypoints_1[valid_correspondence]).mean()

                # print(valid.sum())
                # image = plot_matches(rgb_0, obj_points0, rgb_1, obj_points1, num_points_to_plot=-1)
                # plot_image(image, mode='TkAgg')

                # Estimate the pose

                # Get the xyz position of every valid keypoint
                p_frame_0 = backproject_pixels(pixel_indices=obj_points0, intrinsic_matrix=K,
                                                depth_map=depth_0, depth_units='mm')
                p_frame_1 = backproject_pixels(pixel_indices=obj_points1, intrinsic_matrix=K,
                                                depth_map=depth_1, depth_units='mm')

                # Reshape to (batch_size, 3, num_points) for compatibility with the SVD head
                p_frame_0 = torch.tensor(p_frame_0, dtype=torch.float32).permute(1, 0).unsqueeze(0)
                p_frame_1 = torch.tensor(p_frame_1, dtype=torch.float32).permute(1, 0).unsqueeze(0)

                R_pred, t_pred = svd(p_frame_0, p_frame_1)

                rot_error = compute_rot_error(R_pred=R_pred, R_target=T_delta[..., :3, :3], degrees=True)
                trans_error = compute_translation_error(t_pred=t_pred, t_target=T_delta[..., :3, 3], mm=True)
                print(
                    f'Number of correspondences: {valid.sum()} -- Rotation error: {rot_error.item():.2f} degrees -- Translation error: {trans_error.item():.2f} mm')

                results = pd.concat((results,
                                        pd.DataFrame(
                                            [[int(valid.sum()), interpolation_method, inference_time, mean_pixel_error,
                                            rot_error.item(), trans_error.item()]],
                                            columns=['Number of Correspondences',
                                                    'Interpolation method',
                                                    'Time',
                                                    'Mean correspondence estimation error [pixels]',
                                                    'Rotation error [deg]',
                                                    'Translation Error [mm]'])))

                results.to_csv(f'/home/kamil/dino_results_unmasked_inputs_threshold_{thresh}.csv')










import os
import sys
import time

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import torch
import socket
from tqdm import tqdm

sys.path.append('/home/kamil/phd/sim2real')

from dino_vit_features.correspondences import find_correspondences
from head_cam.data.utils import load_scene, estimate_correspondences, get_keypoint_indices, backproject_pixels
from head_cam.globals import ASSETS_DIR, SCENE_SAVE_FORMAT
from head_cam.error_minimisation.svd_head import SVDHead
from head_cam.error_minimisation.utils import compute_rot_error, compute_translation_error
from sim2real.camera_utils import calculate_intrinsic_for_new_resolution
from dino_vit_features.extractor import ViTExtractor
from PIL import Image


def get_common_objs(instance_attribute_map_0, instance_attribute_map_1):
    objs_1 = [instance['name'] for instance in instance_attribute_map_1 if instance['name'] != 'Floor']
    return [obj for obj in instance_attribute_map_0 if obj['name'] in objs_1]


def sample_random_obj(instance_attribute_map_0, instance_attribute_map_1):
    common_objs = get_common_objs(data['instance_attribute_map_0'], data['instance_attribute_map_1'])
    obj = np.random.choice(common_objs)
    name = obj['name']
    idx = obj['idx']
    return name, idx


if __name__ == '__main__':

    num_pairs = 100000
    load_size = 224
    layer = 9
    facet = 'key'
    bin = True
    thresh = 0.01
    model_type = 'dino_vits8'  # [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
    stride = 4
    svd = SVDHead()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino_extractor = ViTExtractor(model_type, stride, device=device)

    print(f'Threshold used: {thresh}')

    if socket.gethostname() == 'slifer':
        dataset_name = 'main_slifer'
    elif socket.gethostname() == 'omen':
        dataset_name = 'test'

    results = pd.DataFrame(
        columns=['Number of Correspondences', 'Interpolation method', 'Time',
                 'Mean correspondence estimation error [pixels]',
                 'Rotation error [deg]', 'Translation Error [mm]'])

    total_num_objects = 0
    num_objects_with_correspondences = 0

    for scene_idx in tqdm(range(0, 23)):

        scene_dir = os.path.join(ASSETS_DIR, 'datasets', dataset_name, SCENE_SAVE_FORMAT.format(scene_idx))

        # Get all the data for current scene
        data = load_scene(scene_dir, load_T_WC=True, load_rgb=True, load_depth=True, load_intrinsic=True,
                          load_T_CO=True, load_T_WO=True, load_target_obj_segmaps=True, load_instance_segmaps=True,
                          load_instance_attribute_maps=True)

        K = data['intrinsic'].copy()
        rgb_0 = data['rgb_0'].copy()
        rgb_1 = data['rgb_1'].copy()
        all_objs_T_CO_0 = data['T_CO_0'].copy()
        all_objs_T_CO_1 = data['T_CO_1'].copy()

        # Preprocess VIT inputs
        image0_batch, image0_pil = dino_extractor.preprocess_pil_image(Image.fromarray(rgb_0).convert('RGB'),
                                                                       load_size=load_size)
        image1_batch, image1_pil = dino_extractor.preprocess_pil_image(Image.fromarray(rgb_1).convert('RGB'),
                                                                       load_size=load_size)

        # Get desired input size, resize other images and adapt intrinsic matrix
        _, _, inference_height, inference_width = image0_batch.shape

        K = calculate_intrinsic_for_new_resolution(K, new_width=inference_width, new_height=inference_height,
                                                   old_width=data['depth_0'].shape[1],
                                                   old_height=data['depth_0'].shape[0])

        rgb_0 = np.asarray(image0_pil)
        rgb_1 = np.asarray(image1_pil)

        start = time.time()

        try:
            with torch.no_grad():
                points0, points1 = find_correspondences(extractor=dino_extractor,
                                                        image1_batch=image0_batch,
                                                        image2_batch=image1_batch,
                                                        num_pairs=num_pairs,
                                                        layer=layer,
                                                        facet=facet,
                                                        bin=bin,
                                                        thresh=thresh,
                                                        device=device)
            inference_time = time.time() - start
        except Exception as e:
            print(e)
            continue

        # import matplotlib.pyplot as plt
        # fig0, fig1 = draw_correspondences(points0, points1, image0_pil, image1_pil)
        # plt.show()

        points0 = np.asarray(points0)
        points0 = np.concatenate((points0[:, 1:2], points0[:, 0:1]), axis=1)

        points1 = np.asarray(points1)
        points1 = np.concatenate((points1[:, 1:2], points1[:, 0:1]), axis=1)

        for interpolation_method in ['nearest neighbour', 'linear interpolation', 'cubic interpolation',
                                     'area interpolation']:
            depth_0 = data['depth_0'].copy()
            depth_1 = data['depth_1'].copy()
            instance_segmap_0 = data['instance_segmap_0'].copy()
            instance_segmap_1 = data['instance_segmap_1'].copy()

            if interpolation_method == 'nearest neighbour':
                depth_0 = cv2.resize(depth_0, (inference_width, inference_height), 0, 0,
                                     interpolation=cv2.INTER_NEAREST)
                depth_1 = cv2.resize(depth_1, (inference_width, inference_height), 0, 0,
                                     interpolation=cv2.INTER_NEAREST)
            elif interpolation_method == 'linear interpolation':
                depth_0 = cv2.resize(depth_0, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_LINEAR)
                depth_1 = cv2.resize(depth_1, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_LINEAR)
            elif interpolation_method == 'cubic interpolation':
                depth_0 = cv2.resize(depth_0, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_CUBIC)
                depth_1 = cv2.resize(depth_1, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_CUBIC)
            elif interpolation_method == 'area interpolation':
                depth_0 = cv2.resize(depth_0, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_AREA)
                depth_1 = cv2.resize(depth_1, (inference_width, inference_height), 0, 0, interpolation=cv2.INTER_AREA)

            instance_segmap_0 = cv2.resize(instance_segmap_0, (inference_width, inference_height), 0, 0,
                                           interpolation=cv2.INTER_NEAREST)
            instance_segmap_1 = cv2.resize(instance_segmap_1, (inference_width, inference_height), 0, 0,
                                           interpolation=cv2.INTER_NEAREST)

            common_objs = get_common_objs(data['instance_attribute_map_0'], data['instance_attribute_map_1'])

            for obj in common_objs:
                name, idx = obj['name'], obj['idx']

                obj_segmap_0 = instance_segmap_0 == idx
                obj_segmap_1 = instance_segmap_1 == idx
                T_CO_0 = all_objs_T_CO_0[name]
                T_CO_1 = all_objs_T_CO_1[name]

                # Ensure that the object is fully visible
                if obj_segmap_0[:, 0].any() or \
                        obj_segmap_0[:, -1].any() or \
                        obj_segmap_0[0, :].any() or \
                        obj_segmap_0[-1, :].any() or \
                        obj_segmap_1[:, 0].any() or \
                        obj_segmap_1[:, -1].any() or \
                        obj_segmap_1[0, :].any() or \
                        obj_segmap_1[-1, :].any():
                    continue

                if interpolation_method == 'nearest neighbour':
                    total_num_objects += 1

                T_delta = torch.tensor(T_CO_1 @ np.linalg.inv(T_CO_0), dtype=torch.float32).unsqueeze(0)

                # image = plot_matches(rgb_0, points0, rgb_1, points1, num_points_to_plot=-1)
                # plot_image(image, mode='TkAgg')

                obj_indices_0 = get_keypoint_indices(obj_segmap_0)
                obj_indices_1 = get_keypoint_indices(obj_segmap_1)

                valid_0 = np.zeros(len(points0), dtype=bool)

                for i, kpt in enumerate(points0):
                    if (kpt == obj_indices_0).all(axis=1).any():
                        valid_0[i] = True

                valid_1 = np.zeros(len(points1), dtype=bool)

                for i, kpt in enumerate(points1):
                    if (kpt == obj_indices_1).all(axis=1).any():
                        valid_1[i] = True

                valid = np.logical_and(valid_0, valid_1)

                if valid.sum() > 3:
                    if interpolation_method == 'nearest neighbour':
                        num_objects_with_correspondences += 1

                    obj_points0 = points0[valid]
                    obj_points1 = points1[valid]

                    # Get error in the keypoint locations: TODO
                    keypoints_1, valid_correspondence = estimate_correspondences(obj_points0,
                                                                                 depth_0,
                                                                                 depth_1,
                                                                                 T_CO_0,
                                                                                 T_CO_1,
                                                                                 K,
                                                                                 depth_rejection_threshold=0.02,
                                                                                 return_valid_list=True,
                                                                                 depth_units='mm')

                    mean_pixel_error = np.abs(
                        obj_points1[valid_correspondence] - keypoints_1[valid_correspondence]).mean()

                    # print(valid.sum())
                    # image = plot_matches(rgb_0, obj_points0, rgb_1, obj_points1, num_points_to_plot=-1)
                    # plot_image(image, mode='TkAgg')

                    # Estimate the pose

                    # Get the xyz position of every valid keypoint
                    p_frame_0 = backproject_pixels(pixel_indices=obj_points0, intrinsic_matrix=K,
                                                   depth_map=depth_0, depth_units='mm')
                    p_frame_1 = backproject_pixels(pixel_indices=obj_points1, intrinsic_matrix=K,
                                                   depth_map=depth_1, depth_units='mm')

                    # Reshape to (batch_size, 3, num_points) for compatibility with the SVD head
                    p_frame_0 = torch.tensor(p_frame_0, dtype=torch.float32).permute(1, 0).unsqueeze(0)
                    p_frame_1 = torch.tensor(p_frame_1, dtype=torch.float32).permute(1, 0).unsqueeze(0)

                    R_pred, t_pred = svd(p_frame_0, p_frame_1)

                    rot_error = compute_rot_error(R_pred=R_pred, R_target=T_delta[..., :3, :3], degrees=True)
                    trans_error = compute_translation_error(t_pred=t_pred, t_target=T_delta[..., :3, 3], mm=True)
                    print(
                        f'Number of correspondences: {valid.sum()} -- Rotation error: {rot_error.item():.2f} degrees -- Translation error: {trans_error.item():.2f} mm')

                    results = pd.concat((results,
                                         pd.DataFrame(
                                             [[int(valid.sum()), interpolation_method, inference_time, mean_pixel_error,
                                               rot_error.item(), trans_error.item()]],
                                             columns=['Number of Correspondences',
                                                      'Interpolation method',
                                                      'Time',
                                                      'Mean correspondence estimation error [pixels]',
                                                      'Rotation error [deg]',
                                                      'Translation Error [mm]'])))

                    results.to_csv(f'/home/kamil/dino_results_unmasked_inputs_threshold_{thresh}.csv')

    print(
        f'\n\nNumber of objects fully in the image for which >3 correspondences were found: {num_objects_with_correspondences} \ {total_num_objects}')
