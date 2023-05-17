import glob
import json
import os
import math
import msgpack
import msgpack_numpy as m

from loguru import logger

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
from .utils import (rgb2gray, pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, calculate_rot_delta,
                    get_keypoint_indices, plot_matches, estimate_cropped_correspondences, calculate_intrinsic_for_new_resolution)
from .preprocessing import resize_img_pair

from .debug_utils import estimate_correspondences, estimate_correspondences_diff_intr

m.patch()
# ORIGINAL_IMAGE_WIDTH = 640
# ORIGINAL_IMAGE_HEIGHT = 480
# RENDERING_IMAGE_WIDTH = 640
# RENDERING_IMAGE_HEIGHT = 480
# RENDERING_RS_IMAGE_WIDTH = 64
# RENDERING_RS_IMAGE_HEIGHT = 48
# ORIGINAL_INTRINSIC = np.array([[612.044, 0, 326.732],
#                                [0, 611.178, 228.342],
#                                [0, 0, 1]])

# RENDERING_INTRINSIC = calculate_intrinsic_for_new_resolution(
#     ORIGINAL_INTRINSIC, RENDERING_IMAGE_WIDTH, RENDERING_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT)

class BlenderDataset(Dataset):

    def __init__(self,
                 use_masks: bool = False,
                 crop_margin: float = 0.3,
                 resize_modality: int = 0,
                 segment_object: bool = False,
                 filter_data: bool = False,
                 coarse_grid_factor = 8) -> None:
        self.dataset_dir = DATASET_DIR
        self.coarse_scale = 0.125 #for training LoFTR
        self.coarse_grid_factor = coarse_grid_factor
        self.idx = None

        self.use_masks = use_masks
        self.crop_margin = crop_margin
        self.resize_modality = resize_modality
        self.segment_object = segment_object
        self.filter_data = filter_data

        self.scene_names = np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*')))

    def __len__(self):
        return len(self.scene_names)

    def load_scene(self, scene_dir) -> dict:
        with open(scene_dir, "rb") as data_file:
            byte_data = data_file.read()
            data: dict = msgpack.unpackb(byte_data)

        # for k in data.keys():
        #     print(k)

        # print(data["colors"].shape)
        # print(data["depth"].shape)
        # print(data["colors"])
        # print(data["depth"])

        data["depth"][0] = data["depth"][0] * (data["depth"][0] < 5.0)
        data["depth"][1] = data["depth"][1] * (data["depth"][1] < 5.0)

        # print(data["depth"][0].dtype)
        # # print(data["depth"][0])
        # print(np.min(data["depth"][0]), np.max(data["depth"][0]))
        # # plt.figure()
        # # plt.imshow(data["colors"][0])
        # plt.figure()
        # plt.imshow(data["depth"][0])
        # plt.show()
        
        data.update({
            "rgb_0": data["colors"][0],
            "rgb_1": data["colors"][1],
            "depth_0": data["depth"][0] * data["cp_main_obj_segmaps"][0] * 1000,
            "depth_1": data["depth"][1] * data["cp_main_obj_segmaps"][1] * 1000
        })
        data.pop("colors")
        data.pop("depth")
        return data
    
    def load_random_scene(self):
        self.idx = np.random.randint(0, len(self))
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        return self.load_scene(scene_dir)
    
    def check_matches(self, crop_data, T_01):
        obj_indices_0 = get_keypoint_indices(crop_data["seg_0"], self.coarse_grid_factor)

        # grid = np.zeros(crop_data["depth_0"].shape)
        # grid[obj_indices_0[:,1], obj_indices_0[:,0]] = 1
        # plt.figure()
        # plt.imshow(grid)
        # plt.show()

        assert isinstance(obj_indices_0, np.ndarray), 'Keypoints must be stored in a numpy array'
        assert obj_indices_0.dtype == np.int64, 'Keypoints should be integers'
        assert len(obj_indices_0.shape) == 2, 'Keypoints must be stored in a 2-dimensional array'
        assert obj_indices_0.shape[1] == 2, 'The x and y position of all keypoints must be specified'
        matches_data = estimate_cropped_correspondences(obj_indices_0,
                                                        crop_data["depth_0"].copy(),
                                                        crop_data["depth_1"].copy(),
                                                        T_01.copy(),
                                                        crop_data["intrinsics_0"].copy(),
                                                        crop_data["intrinsics_1"].copy(),
                                                        depth_rejection_threshold=0.003,
                                                        depth_units='mm')
        crop_data["keypoints_1"], crop_data["valid_correspondence"] = matches_data[:,:2], matches_data[:,2].astype(bool)

        # plt.figure()
        # plt.imshow(crop_data["depth_0"] * crop_data["seg_0"])
        # # plt.imshow(crop_data["depth_0"])
        # plt.figure()
        # plt.imshow(crop_data["gray_1"][0] * crop_data["seg_1"])

        filtered_keypoints_0 = obj_indices_0[crop_data["valid_correspondence"]]
        filtered_keypoints_1 = crop_data["keypoints_1"][crop_data["valid_correspondence"]]
        # plot_matches(
        #     crop_data["gray_0"] * crop_data["seg_0"],
        #     filtered_keypoints_0,
        #     crop_data["gray_1"] * crop_data["seg_1"],
        #     filtered_keypoints_1,
        #     num_points_to_plot=-1,
        #     shuffle_matches=True
        # )
    
    def get_common_objs(self, instance_attribute_map_0, instance_attribute_map_1):
        objs_1 = [instance['name'] for instance in instance_attribute_map_1 if instance['name'] != 'Floor']
        return [obj for obj in instance_attribute_map_0 if obj['name'] in objs_1]

    def crop_object(self, data: dict):
        crop_data = {}

        rgb0 = data["rgb_0"].copy()
        depth0 = data["depth_0"].copy()
        segmap0 = data["cp_main_obj_segmaps"][0].copy()

        rgb1 = data["rgb_1"].copy()
        depth1 = data["depth_1"].copy()
        segmap1 = data["cp_main_obj_segmaps"][1].copy()

        bbox0 = bbox_from_mask(segmap0, margin=self.crop_margin)
        bbox1 = bbox_from_mask(segmap1, margin=self.crop_margin)

        rgb0, crop_data["depth_0"], crop_data["seg_0"], bbox0 = crop(bbox0, rgb0, depth0, segmap0)
        rgb1, crop_data["depth_1"], crop_data["seg_1"], bbox1 = crop(bbox1, rgb1, depth1, segmap1)

        crop_data["gray_0"] = rgb2gray(rgb0) / 255
        crop_data["gray_1"] = rgb2gray(rgb1) / 255

        # crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
        #     RENDERING_INTRINSIC.copy(), top=bbox0[1], left=bbox0[0]
        # )
        # crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
        #     RENDERING_INTRINSIC.copy(), top=bbox1[1], left=bbox1[0]
        # )
        crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox0[1], left=bbox0[0]
        )
        crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
            data["intrinsic"].copy(), top=bbox1[1], left=bbox1[0]
        )

        resize_img_pair(crop_data, self.resize_modality)

        return crop_data

    def get_rel_transformation(self, data):
        # T_C0 = pose_inv(data["T_WC_opencv"]) @ data["T_WO_frame_0"]
        # T_1C = pose_inv(data["T_WO_frame_1"]) @ data["T_WC_opencv"]
        # T_C0C1 = T_C0 @ T_1C
        # T_delta = T_C0 @ T_1C
        # return T_C0C1, pose_inv(T_C0C1), T_delta
        T_C0 = pose_inv(data["T_WC_opencv"]) @ data["T_WO_frame_0"]
        T_1C = pose_inv(data["T_WO_frame_1"]) @ data["T_WC_opencv"]
        T_C0C1 = T_C0 @ T_1C
        return T_C0C1, pose_inv(T_C0C1)

    def __getitem__(self, idx):
        # Check length of Dataset is respected
        # self.idx = np.random.randint(0,2)
        self.idx = idx
        if self.idx >= len(self):
            raise IndexError

        object_is_visible = False
        while not object_is_visible:
            try:
                # Get all the data for current scene
                scene_dir = self.scene_names[self.idx]
                data = self.load_scene(scene_dir)
                crop_data = self.crop_object(data)
                T_01, T_10 = self.get_rel_transformation(data)
                self.check_matches(crop_data, T_01)
                if np.sum(crop_data["valid_correspondence"]) >= 200: ##############################
                    object_is_visible = True
                else:
                    self.idx = np.random.randint(0, len(self))
            except Exception as e:
                logger.warning(f"The following exception was found: \n{e}")
                self.idx = np.random.randint(0, len(self))

        if self.segment_object:
            crop_data["gray_0"] *= crop_data["seg_0"]
            crop_data["depth_0"] *= crop_data["seg_0"]
            crop_data["gray_1"] *= crop_data["seg_1"]
            crop_data["depth_1"] *= crop_data["seg_1"]
        
        crop_data["depth_0"] = crop_data["depth_0"].astype(np.float16) / 1e3
        crop_data["depth_1"] = crop_data["depth_1"].astype(np.float16) / 1e3

        data = {
            'image0': crop_data["gray_0"].astype(np.float32),   # (1, h, w)
            'depth0': crop_data["depth_0"],   # (h, w)
            'image1': crop_data["gray_1"].astype(np.float32),
            'depth1': crop_data["depth_1"], # NOTE: maybe int32?
            'T_0to1': T_10.astype(np.float32),   # (4, 4)
            'T_1to0': T_01.astype(np.float32),
            'K0': crop_data["intrinsics_0"].astype(np.float32),  # (3, 3)
            'K1': crop_data["intrinsics_1"].astype(np.float32),
            'dataset_name': 'Blender',
            'scene_id': self.idx,
            'pair_id': 0,
            'pair_names': (f"scene_{self.idx}_0",
                           f"scene_{self.idx}_1")
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

    print(f"\nThe dataset length is: {len(dataset)}")

    while True:
        for point in dataset:
            print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")

    # for point in dataset:
    #     print("\n")
    #     for key in point.keys():
    #         if isinstance(point[key], np.ndarray):
    #             tp = point[key].dtype
    #         else:
    #             tp = type(point[key])
    #         print(f"{key}: {tp}")

    dataset[2]

    # print(len(dataset))

    # while True:
    #     for i in range(len(dataset)):
    #         dataset[i]
    #     print("\n")