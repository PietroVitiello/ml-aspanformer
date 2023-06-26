import glob
import json
import os
import math
from copy import deepcopy

import msgpack
import msgpack_numpy as m

from loguru import logger
from os import path as osp
from typing import Dict
from unicodedata import name
import matplotlib.pyplot as plt
import open3d as o3d

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from .globals import REALWORLD_DATA_DIR
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

class RealWorldDataset(Dataset):

    def __init__(self,
                 use_masks: bool = False,
                 crop_margin: float = 0.3,
                 resize_modality: int = 5,
                 segment_object: bool = True,
                 filter_data: bool = False,
                 coarse_grid_factor = 8) -> None:
        self.dataset_dir = REALWORLD_DATA_DIR
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

    def load_scene(self, scene_filename) -> dict:
        with open(scene_filename, "rb") as data_file:
            byte_data = data_file.read()
            data: dict = msgpack.unpackb(byte_data)

        data = deepcopy(data)
        # kernel = np.ones((5, 5), dtype=np.float32)
        kernel = np.ones((5,5), dtype=np.float32) / 25
        # # # kernel = np.ones((2,2), dtype=np.float32) / 4

        # # cv2.imshow("Normal", data["cp_main_obj_segmaps"][0].astype(np.float32))

        data["seg_0"] = cv2.filter2D(data["cp_main_obj_segmaps"][0].astype(np.float32), -1, kernel)
        data["seg_1"] = cv2.filter2D(data["cp_main_obj_segmaps"][1].astype(np.float32), -1, kernel)
        data["seg_0"] = (data["seg_0"] >= 1)
        data["seg_1"] = (data["seg_1"] >= 1)

        # print(data["seg_0"].shape)
        # print(data["seg_0"].dtype)

        # cv2.imshow("Eroded", data["seg_0"].astype(np.float32))
        # cv2.waitKey(0)

        # data["seg_0"] = data["cp_main_obj_segmaps"][0].astype(bool)
        # data["seg_1"] = data["cp_main_obj_segmaps"][1].astype(bool)

        data["depth"][0] = data["depth"][0]
        data["depth"][1] = data["depth"][1]

        data["depth"][0] = data["depth"][0] * (data["depth"][0] < 5000) * data["seg_0"]
        data["depth"][1] = data["depth"][1] * (data["depth"][1] < 5000) * data["seg_1"]

        # print(data["depth"][1])
        # print(data["depth"][1].max())
        # plt.figure()
        # plt.imshow(data["depth"][1])
        # plt.show()
        
        data.update({
            "rgb_0": data["colors"][0].astype(np.float32),
            "rgb_1": data["colors"][1].astype(np.float32),
            "depth_0": data["depth"][0].astype(np.float32),
            "depth_1": data["depth"][1].astype(np.float32)
        })

        data.pop("colors")
        data.pop("depth")
        data.pop("cp_main_obj_segmaps")

        # plt.figure()
        # plt.imshow(data["rgb_1"]/255)
        # plt.figure()
        # plt.imshow(data["depth_1"])
        # plt.show()
        
        return data
    
    def load_random_scene(self):
        self.idx = np.random.randint(0, len(self))
        scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
        return self.load_scene(scene_dir)
    
    def get_common_objs(self, instance_attribute_map_0, instance_attribute_map_1):
        objs_1 = [instance['name'] for instance in instance_attribute_map_1 if instance['name'] != 'Floor']
        return [obj for obj in instance_attribute_map_0 if obj['name'] in objs_1]
    
    def get_pointcloud_centre(self, depth, intrinsic_matrix):
        depth = (depth * 1000).astype(np.uint16)
        intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix
        depth_o3d = o3d.geometry.Image(depth.copy())
        pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                    extrinsic=np.eye(4))
        return np.mean(np.array(pcd_o3d.points), axis=0)
        
    def crop_object(self, data: dict):
        crop_data = {}

        rgb0 = data["rgb_0"].copy()
        depth0 = data["depth_0"].copy()
        segmap0 = data["seg_0"].copy()

        rgb1 = data["rgb_1"].copy()
        depth1 = data["depth_1"].copy()
        segmap1 = data["seg_1"].copy()

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
        T_WC = data["T_WC_opencv"]
        T_CW = pose_inv(T_WC)
        T_C1 = T_CW @ data["T_WO_frame_1"]
        T_0C = pose_inv(data["T_WO_frame_0"]) @ T_WC
        T_delta_cam = T_C1 @ T_0C
        # T_delta_base = T_WC @ T_delta_cam @ T_CW
        return T_delta_cam
    
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
                                                        crop_data["depth_0"].copy().astype(np.float32),
                                                        crop_data["depth_1"].copy().astype(np.float32),
                                                        T_01.copy(),
                                                        crop_data["intrinsics_0"].copy(),
                                                        crop_data["intrinsics_1"].copy(),
                                                        depth_rejection_threshold=0.005,
                                                        depth_units='m')
        crop_data["keypoints_1"], crop_data["valid_correspondence"] = matches_data[:,:2], matches_data[:,2].astype(bool)

        # plt.figure()
        # plt.imshow(crop_data["depth_0"] * crop_data["seg_0"])
        # # plt.imshow(crop_data["depth_0"])
        # plt.figure()
        # plt.imshow(crop_data["gray_1"][0] * crop_data["seg_1"])

        filtered_keypoints_0 = obj_indices_0 #[crop_data["valid_correspondence"]]
        filtered_keypoints_1 = crop_data["keypoints_1"] #[crop_data["valid_correspondence"]]
        plot_matches(
            crop_data["gray_0"],
            filtered_keypoints_0,
            crop_data["gray_1"],
            filtered_keypoints_1,
            num_points_to_plot=-1,
            shuffle_matches=True
        )

    def __getitem__(self, idx):
        # Check length of Dataset is respected
        # self.idx = np.random.randint(0,2)
        self.idx = idx
        if self.idx >= len(self):
            raise IndexError

        try:
            # Get all the data for current scene
            scene_dir = self.scene_names[self.idx]
            data = self.load_scene(scene_dir)
            crop_data = self.crop_object(data)
            T_delta_wrong_t = self.get_rel_transformation(data)        
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

        pcd0_c = self.get_pointcloud_centre(data["depth0"].copy(), data["K0"].copy())
        pcd1_c = self.get_pointcloud_centre(data["depth1"].copy(), data["K1"].copy())
        T_delta = T_delta_wrong_t.copy()
        T_delta[:3,3] = pcd1_c - T_delta[:3,:3] @ pcd0_c
        data.update({'T_apriltag_gt': T_delta})

        # self.check_matches(crop_data, pose_inv(T_delta))

        return data

if __name__ == "__main__":
    import open3d as o3d

    def img_to_o3d_pcd(depth: np.ndarray, intrinsic_matrix: np.ndarray, rgb=None):
        if depth.dtype == np.int32:
            depth = depth.astype(np.uint16)
        assert depth.dtype == np.uint16, f'The depth image must be \'mm\' stored in the dtype np.uint16 and not {depth.dtype}'
        if rgb is not None:
            assert rgb.dtype == np.uint8, f'The RGB image must be stored in the dtype np.uint8 and not {depth.rgb}'

        intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix
        depth_o3d = o3d.geometry.Image(depth.copy())
        if rgb is not None:
            rgb_o3d = o3d.geometry.Image(rgb.copy())
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_o3d, depth=depth_o3d,
                                                                        depth_scale=1000,
                                                                        convert_rgb_to_intensity=False)
            pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                    extrinsic=np.eye(4))
        else:
            pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                    extrinsic=np.eye(4))
        return pcd_o3d

    dataset = RealWorldDataset(use_masks=True, resize_modality=5)

    print(f"\nThe dataset length is: {len(dataset)}")

    while True:
        for point in dataset:
            print(f"Scene id: {point['scene_id']}\nObject id: {point['pair_id']}\n")

            # depth = (point['depth0'] * 1000).astype(np.uint16)
            # pcd1 = img_to_o3d_pcd(depth, point['K0'])
            # pcd1.paint_uniform_color([1, 0.706, 0])
            # depth = (point['depth1'] * 1000).astype(np.uint16)
            # pcd2 = img_to_o3d_pcd(depth, point['K1'])
            # pcd2.paint_uniform_color([0, 0.651, 0.929])
            # pcd1.transform(point['T_apriltag_gt'])
            # o3d.visualization.draw([pcd1, pcd2])

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