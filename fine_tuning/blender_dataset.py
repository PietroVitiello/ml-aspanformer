import glob
import json
import os
import math
import msgpack
import msgpack_numpy as m

import copy

from os import path as osp
from typing import Dict
from unicodedata import name
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from .globals import DATASET_DIR
from .utils import rgb2gray, pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, get_keypoint_indices, plot_matches
from .preprocessing import resize_img_pair

from .debug_utils import estimate_correspondences, estimate_cropped_correspondences, estimate_correspondences_diff_intr

m.patch()

class BlenderDataset(Dataset):

    def __init__(self,
                 use_masks: bool = False,
                 crop_margin: float = 0.3,
                 resize_modality: int = 0) -> None:
        self.dataset_dir = DATASET_DIR
        self.use_masks = use_masks
        self.crop_margin = crop_margin
        self.resize_modality = resize_modality
        self.coarse_scale = 0.125 #for training LoFTR

    def __len__(self):
        return len(np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*'))))
        # return len(next(os.walk(self.dataset_dir))[1])
    
    def choose_object(self, data):
        attributes = data["instance_attribute_map_0"]
        chosen_id = np.random.choice(data['available_ids'])

        # chosen_id = 9

        data['chosen_obj_name'] = next((
            item['name'] for item in attributes if item['idx'] == chosen_id), None)
        data['seg_map_chosen_arg'] = chosen_id
        return data

    # def load_scene(self, scene_dir) -> dict:
                
    #     data = {}
    #     n = len(glob.glob(os.path.join(scene_dir, 'colors', '*')))

    #     # load_intrinsic
    #     data['intrinsic'] = np.load(os.path.join(scene_dir, 'intrinsic.npy'))
    #     # load_T_CO
    #     for i in range(n):
    #         with open(os.path.join(scene_dir, 'T_CO', f'{i}.json')) as jsonfile:
    #             T_WO = json.load(jsonfile)
    #         for obj_name in T_WO:
    #             T_WO[obj_name] = np.asarray(T_WO[obj_name])
    #         data[f'T_CO_{i}'] = T_WO
    #     # load_rgb
    #     for i in range(n):
    #         data[f'rgb_{i}'] = np.asarray(Image.open(os.path.join(scene_dir, 'colors', f'{i}' + '.png')))
    #     # load_depth
    #     for i in range(n):
    #         data[f'depth_{i}'] = np.asarray(
    #             Image.open(os.path.join(scene_dir, 'depth', f'{i}' + '.png'))).astype(np.uint16)
    #     # load_instance_segmaps
    #     for i in range(n):
    #         data[f'instance_segmap_{i}'] = np.asarray(
    #             Image.open(os.path.join(scene_dir, 'instance_segmaps', f'{i}' + '.png')))
    #     # load_instance_attribute_maps
    #     for i in range(n):
    #         with open(os.path.join(scene_dir, 'instance_attribute_maps', f'{i}.json')) as f:
    #             data[f'instance_attribute_map_{i}'] = json.load(f)

    #     data = self.choose_object(data)
    #     return data

    def load_scene(self, scene_dir) -> dict:
        scene_filename = scene_dir + '.msgpack'
        with open(scene_filename, "rb") as data_file:
            byte_data = data_file.read()
            data = msgpack.unpackb(byte_data)
        data['available_ids']: list = np.unique(data["instance_segmap_0"]).tolist()
        data['available_ids'].remove(2) # remove the floor id
        data = self.choose_object(data)
        return data
    
    def get_common_objs(self, instance_attribute_map_0, instance_attribute_map_1):
        objs_1 = [instance['name'] for instance in instance_attribute_map_1 if instance['name'] != 'Floor']
        return [obj for obj in instance_attribute_map_0 if obj['name'] in objs_1]

    def crop_object(self, data: dict):
        crop_data = {}
        obj_id = data['seg_map_chosen_arg']

        rgb0 = data["rgb_0"].copy()
        depth0 = data["depth_0"].copy()
        segmap0 = data["instance_segmap_0"].copy() == obj_id
        # segmap0 = segmap0 == 0

        rgb1 = data["rgb_1"].copy()
        depth1 = data["depth_1"].copy()
        segmap1 = data["instance_segmap_1"].copy() == obj_id
        # segmap1 = segmap1 == 0

        bbox0 = bbox_from_mask(segmap0, margin=self.crop_margin)
        bbox1 = bbox_from_mask(segmap1, margin=self.crop_margin)

        # if self.use_masks:
        #     rgb0, crop_data["depth_0"], crop_data["seg_0"], bbox0 = crop(bbox0, rgb0, depth0, segmap0, return_updated_bbox=True)
        #     rgb1, crop_data["depth_1"], crop_data["seg_1"], bbox1 = crop(bbox1, rgb1, depth1, segmap1, return_updated_bbox=True)
        # else:
        #     rgb0, crop_data["depth_0"], bbox0 = crop(bbox0, rgb0, depth0, return_updated_bbox=True)
        #     rgb1, crop_data["depth_1"], bbox1 = crop(bbox1, rgb1, depth1, return_updated_bbox=True)
        rgb0, crop_data["depth_0"], crop_data["seg_0"], bbox0 = crop(bbox0, rgb0, depth0, segmap0, return_updated_bbox=True)
        rgb1, crop_data["depth_1"], crop_data["seg_1"], bbox1 = crop(bbox1, rgb1, depth1, segmap1, return_updated_bbox=True)

        crop_data["gray_0"] = rgb2gray(rgb0) / 255
        crop_data["gray_1"] = rgb2gray(rgb1) / 255
        # crop_data["depth_0"] = crop_data["depth_0"].astype(np.float16) / 1e3
        # crop_data["depth_1"] = crop_data["depth_1"].astype(np.float16) / 1e3
        crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox0[1], left=bbox0[0]
        )
        crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox1[1], left=bbox1[0]
        )

        resize_img_pair(crop_data, self.resize_modality)

        # plt.figure()
        # plt.imshow(crop_data["seg_0"])

        # plt.figure()
        # plt.imshow(segmap0)
        # plt.figure()
        # plt.imshow(crop_data["gray_0"].transpose((1, 2, 0)))
        # plt.figure()
        # plt.imshow(crop_data["gray_1"].transpose((1, 2, 0)))
        # plt.figure()
        # plt.imshow(crop_data["seg_0"])
        # plt.show()

        return crop_data

    def get_rel_transformation(self, data):
        obj_name = data['chosen_obj_name']
        T_C0 = data["T_CO_0"].get(obj_name)
        T_1C = pose_inv(data["T_CO_1"].get(obj_name))
        T_01 = T_C0 @ T_1C
        return T_01, pose_inv(T_01)

    def __getitem__(self, idx):
        # Get all the data for current scene
        # print(idx)
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(idx).zfill(7)}")
        data = self.load_scene(scene_dir)

        object_is_visible = False
        while not object_is_visible:
            try:
                crop_data = self.crop_object(data)

                T_01, T_10 = self.get_rel_transformation(data)

                obj_indices_0 = get_keypoint_indices(crop_data["seg_0"])
                crop_data["keypoints_1"], crop_data["valid_correspondence"] = estimate_cropped_correspondences(
                                                                                    obj_indices_0,
                                                                                    crop_data["depth_0"].copy(),
                                                                                    crop_data["depth_1"].copy(),
                                                                                    T_01.copy(),
                                                                                    crop_data["intrinsics_0"].copy(),
                                                                                    crop_data["intrinsics_1"].copy(),
                                                                                    depth_rejection_threshold=0.001,
                                                                                    return_valid_list=True,
                                                                                    depth_units='mm'
                                                                            )
                if len(obj_indices_0[crop_data["valid_correspondence"]]) >= 3:
                    object_is_visible = True
                else:
                    # print(data['available_ids'], data['seg_map_chosen_arg'])
                    data['available_ids'].remove(data['seg_map_chosen_arg'])
                    data = self.choose_object(data)
            except ValueError:
                data['available_ids'].remove(data['seg_map_chosen_arg'])
                data = self.choose_object(data)

        # filtered_keypoints_0 = obj_indices_0[crop_data["valid_correspondence"]]
        # filtered_keypoints_1 = crop_data["keypoints_1"][crop_data["valid_correspondence"]]
        # plot_matches(
        #     crop_data["gray_0"],
        #     filtered_keypoints_0,
        #     crop_data["gray_1"],
        #     filtered_keypoints_1,
        #     num_points_to_plot=20,
        #     shuffle_matches=True
        # )
        
        plt.figure()
        plt.imshow(crop_data["depth_0"] * crop_data["seg_0"])
        plt.figure()
        plt.imshow(crop_data["gray_0"][0] * crop_data["seg_0"])
        # plt.show()
        
        # crop_data["depth_0"] = crop_data["depth_0"].astype(np.float16) / 1e3
        # crop_data["depth_1"] = crop_data["depth_1"].astype(np.float16) / 1e3
        crop_data["depth_0"] = crop_data["depth_0"].astype(np.float16) * crop_data["seg_0"]
        crop_data["depth_1"] = crop_data["depth_1"].astype(np.float16) * crop_data["seg_1"]
        
        data = {
            'image0': crop_data["gray_0"].astype(np.float32),   # (1, h, w)
            'depth0': crop_data["depth_0"],   # (h, w)
            'image1': crop_data["gray_1"].astype(np.float32),
            'depth1': crop_data["depth_1"], # TODO: maybe int32?
            # 'T_0to1': T_0to1.astype(np.float32),   # (4, 4)
            # 'T_1to0': T_1to0.astype(np.float32),
            'T_0to1': T_10.astype(np.float32),   # (4, 4)
            'T_1to0': T_01.astype(np.float32),
            'K0': crop_data["intrinsics_0"].astype(np.float32),  # (3, 3)
            'K1': crop_data["intrinsics_1"].astype(np.float32),
            'dataset_name': 'Blender',
            'scene_id': idx,
            'pair_id': data['seg_map_chosen_arg'],
            'pair_names': (f"scene_{idx}_object_{data['seg_map_chosen_arg']}_0",
                           f"scene_{idx}_object_{data['seg_map_chosen_arg']}_1")
        }
        if self.use_masks:
            crop_data["seg_0"], crop_data["seg_1"] = torch.from_numpy(crop_data["seg_0"]), torch.from_numpy(crop_data["seg_1"])
            [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([crop_data["seg_0"], crop_data["seg_1"]], dim=0)[None].float(),
                                                   scale_factor=self.coarse_scale,
                                                   mode='nearest',
                                                   recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data

if __name__ == "__main__":

    dataset = BlenderDataset(use_masks=True, resize_modality=5)

    for point in dataset:
        print("\n")
        for key in point.keys():
            if isinstance(point[key], np.ndarray):
                tp = point[key].dtype
            else:
                tp = type(point[key])
            print(f"{key}: {tp}")

    dataset[4]

    # print(len(dataset))

    # while True:
    #     for i in range(len(dataset)):
    #         dataset[i]
    #     print("\n")