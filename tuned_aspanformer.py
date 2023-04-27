from pathlib import Path

import torch
import torch.nn as nn

from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config

from src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_Matcher
from fine_tuning.aspan_preprocessor import ASpanFormer_Preprocessor

class ASpanFormer(nn.Module):

    def __init__(self,
                 config_path: Path = Path('configs/aspan/outdoor/aspan_test.py'),
                 weights_path: Path = Path('weights/hpc_multi_1.ckpt'),
                 images_are_normalised: bool=True,
                 cuda_is_available: bool = True
                 ) -> None:
        super().__init__()
        config = get_cfg_defaults()
        config.merge_from_file(config_path)
        _config = lower_config(config)

        self.preprocessor = ASpanFormer_Preprocessor(images_are_normalised)
        self.model = ASpanFormer_Matcher(config=_config['aspan'])
        self.confidure_model(weights_path, cuda_is_available)
        
    def confidure_model(self, weights_path, cuda_is_available):
        pretrained_state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
        self.model.load_state_dict(pretrained_state_dict, strict=False)  #what is strict=false
        if cuda_is_available:
            self.model.cuda()
        self.model.eval()

    def forward(self, data: dict):
        data = self.preprocessor(data)
        with torch.no_grad():   
            # self.model(data, online_resize=True)     #Check online resize
            self.model(data, online_resize=False)     #Check online resize

        # for k in data.keys():
        #     print(k)
        
        corr0 = data['mkpts0_f'].cpu().numpy()
        corr1 = data['mkpts1_f'].cpu().numpy()

        # print(corr0.shape)

        return corr0, corr1
    




























if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import demo.demo_utils as demo_utils

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

    from fine_tuning.globals import DATASET_DIR
    from fine_tuning.utils import rgb2gray, pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, get_keypoint_indices, plot_matches, estimate_cropped_correspondences
    from fine_tuning.preprocessing import resize_img_pair

    from fine_tuning.debug_utils import estimate_correspondences, estimate_correspondences_diff_intr

    m.patch()

    class BlenderDataset(Dataset):

        def __init__(self,
                    use_masks: bool = False,
                    crop_margin: float = 0.3,
                    resize_modality: int = 0,
                    coarse_grid_factor = 8,
                    segment_object: bool = False,
                    filter_data: bool = False) -> None:
            self.dataset_dir = DATASET_DIR
            self.coarse_scale = 0.125 #for training LoFTR
            self.coarse_grid_factor = coarse_grid_factor
            self.idx = None

            self.use_masks = use_masks
            self.crop_margin = crop_margin
            self.resize_modality = resize_modality
            self.segment_object = segment_object
            self.filter_data = filter_data

        def __len__(self):
            return len(np.sort(glob.glob(os.path.join(self.dataset_dir, 'scene_*'))))
            # return len(next(os.walk(self.dataset_dir))[1])
        
        def choose_object(self, data):
            attributes = data["instance_attribute_map_0"]
            valid = False
            while valid == False:
                # To avoid empty scene
                if len(data['available_ids']) >= 1:
                    chosen_id = np.random.choice(data['available_ids'])
                    data['available_ids'].remove(chosen_id)

                    # chosen_id = 6
                    # print(data['available_ids'])
                    # print(chosen_id)

                    seg0 = data["instance_segmap_0"].copy() == chosen_id
                    seg1 = data["instance_segmap_1"].copy() == chosen_id
                    # To avoid non co-visible objects
                    if seg0.any() and seg1.any():
                        valid = True
                else:
                    logger.warning(f"Scene {self.idx} has no objects. Loading random scene.")
                    return self.load_random_scene()

            data['chosen_obj_name'] = next((
                item['name'] for item in attributes if item['idx'] == chosen_id), None)
            data['seg_map_chosen_arg'] = chosen_id
            return data

        def load_scene(self, scene_dir) -> dict:
            scene_filename = scene_dir + '.msgpack'
            with open(scene_filename, "rb") as data_file:
                byte_data = data_file.read()
                data = msgpack.unpackb(byte_data)
            data['available_ids']: list = np.unique(data["instance_segmap_0"]).tolist()
            data['available_ids'].remove(2) # remove the floor id
            data = self.choose_object(data)
            return data
        
        def load_random_scene(self):
            self.idx = np.random.randint(0, len(self))
            scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
            return self.load_scene(scene_dir)
        
        def check_matches(self, crop_data, T_01):
            obj_indices_0 = get_keypoint_indices(crop_data["seg_0"], self.coarse_grid_factor)

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
                                                            depth_rejection_threshold=0.001,
                                                            depth_units='mm')
            crop_data["keypoints_1"], crop_data["valid_correspondence"] = matches_data[:,:2], matches_data[:,2].astype(bool)
        
        def get_common_objs(self, instance_attribute_map_0, instance_attribute_map_1):
            objs_1 = [instance['name'] for instance in instance_attribute_map_1 if instance['name'] != 'Floor']
            return [obj for obj in instance_attribute_map_0 if obj['name'] in objs_1]

        def crop_object(self, data: dict):
            crop_data = {}
            obj_id = data['seg_map_chosen_arg']

            rgb0 = data["rgb_0"].copy()
            depth0 = data["depth_0"].copy()
            segmap0 = data["instance_segmap_0"].copy() == obj_id

            rgb1 = data["rgb_1"].copy()
            depth1 = data["depth_1"].copy()
            segmap1 = data["instance_segmap_1"].copy() == obj_id

            bbox0 = bbox_from_mask(segmap0, margin=self.crop_margin)
            bbox1 = bbox_from_mask(segmap1, margin=self.crop_margin)

            crop_data["rgb0"], crop_data["depth_0"], crop_data["seg_0"], bbox0 = crop(bbox0, rgb0, depth0, segmap0)
            crop_data["rgb1"], crop_data["depth_1"], crop_data["seg_1"], bbox1 = crop(bbox1, rgb1, depth1, segmap1)

            crop_data["gray_0"] = rgb2gray( crop_data["rgb0"]) / 255
            crop_data["gray_1"] = rgb2gray( crop_data["rgb1"]) / 255
            crop_data["intrinsics_0"] = calculate_intrinsic_for_crop(
                data["intrinsic"].copy(), top=bbox0[1], left=bbox0[0]
            )
            crop_data["intrinsics_1"] = calculate_intrinsic_for_crop(
                data["intrinsic"].copy(), top=bbox1[1], left=bbox1[0]
            )

            resize_img_pair(crop_data, self.resize_modality)

            return crop_data

        def get_rel_transformation(self, data):
            obj_name = data['chosen_obj_name']
            T_C0 = data["T_CO_0"].get(obj_name)
            T_1C = pose_inv(data["T_CO_1"].get(obj_name))
            T_01 = T_C0 @ T_1C
            return T_01, pose_inv(T_01)

        def __getitem__(self, idx):
            # Check length of Dataset is respected
            self.idx = idx
            if self.idx >= len(self):
                raise IndexError
            
            # Get all the data for current scene
            scene_dir = os.path.join(self.dataset_dir, f"scene_{str(self.idx).zfill(7)}")
            data = self.load_scene(scene_dir)

            object_is_visible = False
            while not object_is_visible:
                try:
                    crop_data = self.crop_object(data)
                    T_01, T_10 = self.get_rel_transformation(data)
                    
                    self.check_matches(crop_data, T_01)
                    if np.sum(crop_data["valid_correspondence"]) >= 200:
                        object_is_visible = True
                    else:
                        # data['available_ids'].remove(data['seg_map_chosen_arg'])
                        data = self.choose_object(data)
                except ValueError:
                    if len(data['available_ids']) > 1:
                        # data['available_ids'].remove(data['seg_map_chosen_arg'])
                        data = self.choose_object(data)
                    else:
                        logger.warning(f"No valid object in scene {self.idx}")
                        data = self.load_random_scene()
                except Exception as e:
                    # data['available_ids'].remove(data['seg_map_chosen_arg'])
                    data = self.choose_object(data)
                    logger.warning(f"The following exception was found: \n{e}")

            if self.segment_object:
                crop_data["gray_0"] *= crop_data["seg_0"]
                crop_data["depth_0"] *= crop_data["seg_0"]
                crop_data["gray_1"] *= crop_data["seg_1"]
                crop_data["depth_1"] *= crop_data["seg_1"]

            
            crop_data["depth_0"] = crop_data["depth_0"].astype(np.float16) / 1e3
            crop_data["depth_1"] = crop_data["depth_1"].astype(np.float16) / 1e3

            crop_data["image1"] = crop_data["rgb0"]
            crop_data["image2"] = crop_data["rgb1"]
            crop_data["seg1"] = crop_data["seg_0"]
            crop_data["seg2"] = crop_data["seg_1"]

            return crop_data

    dataset = BlenderDataset(use_masks=True, segment_object=False, resize_modality=-1, coarse_grid_factor=1)
    model = ASpanFormer(images_are_normalised=False)
    # print(dataset[0])
    for data in dataset:
        # data["image0"] = data["image0"].transpose(1,2,0)
        # data["image1"] = data["image1"].transpose(1,2,0)
        corr0, corr1 = model(data)

        data = model.preprocessor(data)
        img0 = (data['image0'].cpu().numpy()[0,0,:,:,None] * 255).astype(np.uint8)
        img1 = (data['image1'].cpu().numpy()[0,0,:,:,None] * 255).astype(np.uint8)

        F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
        if mask_F is not None:
            mask_F=mask_F[:,0].astype(bool) 
        else:
            mask_F=np.zeros_like(corr0[:,0]).astype(bool)
        
        # visualize match
        # display = demo_utils.draw_match(img0,img1,corr0,corr1)
        display = demo_utils.draw_match(img0,img1,corr0[mask_F],corr1[mask_F])
        plt.imshow(display)
        plt.show()
