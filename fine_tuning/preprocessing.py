import glob
import json
import os
import math

from typing import TypedDict#

from os import path as osp
from typing import Dict
from unicodedata import name
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from .globals import DATASET_DIR
from .utils import rgb2gray, pose_inv, bbox_from_mask, crop, calculate_intrinsic_for_crop, calculate_intrinsic_for_new_resolution

from .debug_utils import estimate_correspondences, estimate_cropped_correspondences, estimate_correspondences_diff_intr

GrayScale_IMG = np.ndarray
Depth_IMG = np.ndarray
SegMask = np.ndarray
IntrinsicsMatrix = np.ndarray # 3x3

class DataDict(TypedDict):
    gray_0: GrayScale_IMG
    depth_0: Depth_IMG
    seg_0: SegMask
    intrinsics_0: IntrinsicsMatrix
    gray_1: GrayScale_IMG
    depth_1: Depth_IMG
    seg_1: SegMask
    intrinsics_1: IntrinsicsMatrix

def find_minimum_valid_size(max_dim_size: int):
    reduced_dim_in_cnn = max_dim_size / 16
    flattened_reduced_dim = reduced_dim_in_cnn**2
    reminder_after_attention = flattened_reduced_dim % 8
    if reminder_after_attention == 0:
        max_dim_size = max_dim_size 
    else:
        max_dim_size = int(
            16 *(reduced_dim_in_cnn + (4 - (reduced_dim_in_cnn % 4)))
        )
    max_dim_size = max_dim_size if max_dim_size > 256 else 256 
    return tuple([max_dim_size]*2)

def find_padding_borders(img_size, target_size):
    margins = target_size - img_size
    borders = []
    for i, margin in enumerate(margins):
        borders.append(math.floor(margin/2))
        borders.append(borders[-1] + img_size[i])
    return tuple(borders)

def pad_images(imgs, target_imgs, borders, intrinsics):
    top, bottom, left, right = borders

    target_imgs[0][:,top:bottom, left:right] = imgs[0] # grayscale
    target_imgs[1][top:bottom, left:right] = imgs[1] # depth
    if len(imgs) == 3:
        target_imgs[2][top:bottom, left:right] = [2] #segmask

    intrinsics = calculate_intrinsic_for_crop(
        intrinsics, -top, -left
    )
    return *tuple(target_imgs), intrinsics
    # top, bottom, left, right = borders
    # if len(imgs) == 2:
    #     gray, depth = imgs
    #     gray_t, depth_t = target_imgs

    #     gray_t[:,top:bottom, left:right] = gray
    #     depth_t[top:bottom, left:right] = depth
    # elif len(imgs) == 3:
    #     gray, depth, seg = imgs
    #     gray_t, depth_t, seg_t = target_imgs

    #     gray_t[:,top:bottom, left:right] = gray
    #     depth_t[top:bottom, left:right] = depth
    #     seg_t[top:bottom, left:right] = seg

    # intrinsics = calculate_intrinsic_for_crop(
    #     intrinsics, -top, -left
    # )
    # return gray_t, depth_t, intrinsics

def get_resize_modality_name(modality: int):
    if modality == 0:
        return "pad_from_corner"
    elif modality == 1:
        return "pad_zeros_zeros"
    elif modality == 2:
        return "pad_zeros_noise"
    elif modality == 3:
        return "pad_noise_noise"
    elif modality == 4:
        return "pad_and_resize"
    elif modality == 5:
        return "full_resize"
    elif modality == -1:
        return "untouched"
    else:
        raise Exception("[ASpanFormer Preprocessing] The resizing modality has to be an integer in (0-5)")


def resize_img_pair(crop_data: DataDict, modality: int):
    if modality == 0:
        pad_from_corner(crop_data)
    elif modality == 1:
        pad_zeros_zeros(crop_data)
    elif modality == 2:
        pad_zeros_noise(crop_data)
    elif modality == 3:
        pad_noise_noise(crop_data)
    elif modality == 4:
        pad_and_resize(crop_data)
    elif modality == 5:
        full_resize(crop_data)
    elif modality == -1:
        untouched(crop_data)
    else:
        raise Exception("[ASpanFormer Preprocessing] The resizing modality has to be an integer in (0-5)")
    
def untouched(crop_data: DataDict):
    '''
    This resizing modality keeps the images unchanged.
    '''
    crop_data["gray_0"] = crop_data["gray_0"].transpose(2,0,1)
    crop_data["gray_1"] = crop_data["gray_1"].transpose(2,0,1)
    crop_data["seg_0"].astype(bool)
    crop_data["seg_1"].astype(bool)
    return crop_data
    
def pad_from_corner(crop_data: DataDict):
    '''
    This resizing modality pads the two images leaving them in the corner.
    The padding is done with zeros for both the grayscale image and the depth image.
    The size of the final image is set to be the achievable minimum valid size for the model.
    '''
    gray0 = crop_data["gray_0"].copy().transpose(2,0,1)
    gray1 = crop_data["gray_1"].copy().transpose(2,0,1)
    depth0 = crop_data["depth_0"].copy()
    depth1 = crop_data["depth_1"].copy()

    sizes = np.vstack((
        np.expand_dims(gray0.shape, 0),
        np.expand_dims(gray1.shape, 0)
        ))[:,1:]
    max_dim_size = np.max(sizes)

    new_size = (480, 640) #find_minimum_valid_size(max_dim_size)

    crop_data["gray_0"] = np.zeros((1, *new_size), dtype=np.float32)
    crop_data["gray_1"] = np.zeros((1, *new_size), dtype=np.float32)
    crop_data["depth_0"] = np.zeros(new_size, dtype=np.int16)
    crop_data["depth_1"] = np.zeros(new_size, dtype=np.int16)

    crop_data["gray_0"][:, :sizes[0,0], :sizes[0,1]] = gray0
    crop_data["gray_1"][:, :sizes[1,0], :sizes[1,1]] = gray1
    crop_data["depth_0"][:sizes[0,0], :sizes[0,1]] = depth0
    crop_data["depth_1"][:sizes[1,0], :sizes[1,1]] = depth1

    if "seg_0" in crop_data:
        seg0 = crop_data["seg_0"].copy()
        seg1 = crop_data["seg_1"].copy()
        crop_data["seg_0"] = np.zeros(new_size, dtype=bool)
        crop_data["seg_1"] = np.zeros(new_size, dtype=bool)
        crop_data["seg_0"][:sizes[0,0], :sizes[0,1]] = seg0
        crop_data["seg_1"][:sizes[1,0], :sizes[1,1]] = seg1

    return crop_data
    
def pad_zeros_zeros(crop_data: DataDict):
    '''
    This resizing modality pads the two images leaving them in the center.
    The padding is done with zeros for both the grayscale image and the depth image.
    The size of the final image is set to be the achievable minimum valid size for the model.
    '''
    gray0 = crop_data["gray_0"].copy().transpose(2,0,1)
    gray1 = crop_data["gray_1"].copy().transpose(2,0,1)
    depth0 = crop_data["depth_0"].copy()
    depth1 = crop_data["depth_1"].copy()

    sizes = np.vstack((
        np.expand_dims(gray0.shape, 0),
        np.expand_dims(gray1.shape, 0)
        ))[:,1:]
    max_dim_size = np.max(sizes)

    new_size = find_minimum_valid_size(max_dim_size)

    gray_target = np.zeros((1, *new_size), dtype=np.float32)
    depth_target = np.zeros(new_size, dtype=np.int16)

    border0 = find_padding_borders(sizes[0,:], new_size)
    border1 = find_padding_borders(sizes[1,:], new_size)

    if "seg_0" in crop_data:
        seg0 = crop_data["seg_0"].copy()
        seg1 = crop_data["seg_1"].copy()
        crop_data["gray_0"], crop_data["depth_0"], crop_data["seg_0"], \
        crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0, seg0),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data["seg_1"], \
        crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1, seg1),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    else:
        crop_data["gray_0"], crop_data["depth_0"], crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0),
            (gray_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1),
            (gray_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    return crop_data

def pad_zeros_noise(crop_data: DataDict):
    '''
    This resizing modality pads the two images leaving them in the center.
    The padding is done with zeros for both the grayscale image and the depth image.
    The size of the final image is set to be the achievable minimum valid size for the model.
    '''
    gray0 = crop_data["gray_0"].copy().transpose(2,0,1)
    gray1 = crop_data["gray_1"].copy().transpose(2,0,1)
    depth0 = crop_data["depth_0"].copy()
    depth1 = crop_data["depth_1"].copy()

    sizes = np.vstack((
        np.expand_dims(gray0.shape, 0),
        np.expand_dims(gray1.shape, 0)
        ))[:,1:]
    max_dim_size = np.max(sizes)

    new_size = find_minimum_valid_size(max_dim_size)
    
    depth_max, depth_min = np.max(depth0), np.min(depth0)

    gray_target = np.zeros((1, *new_size), dtype=np.float32)
    depth_target = np.random.randint(depth_min, depth_max, new_size, dtype=np.int16)

    border0 = find_padding_borders(sizes[0,:], new_size)
    border1 = find_padding_borders(sizes[1,:], new_size)

    if "seg_0" in crop_data:
        seg0 = crop_data["seg_0"].copy()
        seg1 = crop_data["seg_1"].copy()
        crop_data["gray_0"], crop_data["depth_0"], crop_data["seg_0"], \
        crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0, seg0),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data["seg_1"], \
        crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1, seg1),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    else:
        crop_data["gray_0"], crop_data["depth_0"], crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0),
            (gray_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1),
            (gray_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    return crop_data

def pad_noise_noise(crop_data: DataDict):
    '''
    This resizing modality pads the two images leaving them in the center.
    The padding is done with zeros for both the grayscale image and the depth image.
    The size of the final image is set to be the achievable minimum valid size for the model.
    '''
    gray0 = crop_data["gray_0"].copy().transpose(2,0,1)
    gray1 = crop_data["gray_1"].copy().transpose(2,0,1)
    depth0 = crop_data["depth_0"].copy()
    depth1 = crop_data["depth_1"].copy()

    sizes = np.vstack((
        np.expand_dims(gray0.shape, 0),
        np.expand_dims(gray1.shape, 0)
        ))[:,1:]
    max_dim_size = np.max(sizes)

    new_size = find_minimum_valid_size(max_dim_size)

    gray_max, gray_min = np.max(gray0), np.min(gray0)
    depth_max, depth_min = np.max(depth0), np.min(depth0)

    gray_target = np.random.randint(gray_min, gray_max, (1, *new_size)).astype(np.float32)
    depth_target = np.random.randint(depth_min, depth_max, new_size, dtype=np.int16)

    border0 = find_padding_borders(sizes[0,:], new_size)
    border1 = find_padding_borders(sizes[1,:], new_size)

    if "seg_0" in crop_data:
        seg0 = crop_data["seg_0"].copy()
        seg1 = crop_data["seg_1"].copy()
        crop_data["gray_0"], crop_data["depth_0"], crop_data["seg_0"], \
        crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0, seg0),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data["seg_1"], \
        crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1, seg1),
            (gray_target.copy(), depth_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    else:
        crop_data["gray_0"], crop_data["depth_0"], crop_data['intrinsics_0'] = pad_images(
            (gray0, depth0),
            (gray_target.copy(), depth_target.copy()),
            border0,
            crop_data['intrinsics_0']
        )
        crop_data["gray_1"], crop_data["depth_1"], crop_data['intrinsics_1'] = pad_images(
            (gray1, depth1),
            (gray_target.copy(), depth_target.copy()),
            border1,
            crop_data['intrinsics_1']
        )

    return crop_data

def pad_and_resize(crop_data: DataDict):
    raise NotImplementedError("Pad and Resize have not been implemented yet")

def full_resize(crop_data: DataDict):
    '''
    This resizing modality resizes the two images to the size, preserving their aspect ratios.
    The size of the final image is set to be the achievable minimum valid size for the model.
    '''
    gray0 = crop_data["gray_0"].copy()
    gray1 = crop_data["gray_1"].copy()
    depth0 = crop_data["depth_0"].copy()
    depth1 = crop_data["depth_1"].copy()

    sizes = np.vstack((
        np.expand_dims(gray0.shape, 0),
        np.expand_dims(gray1.shape, 0)
        ))[:,:-1]

    old_sizes = sizes
    max_size = [320, 320] #[240, 320]
    sizes = np.round(sizes * np.min(np.array([max_size]) / sizes, keepdims=True, axis=1)).astype(np.int16)

    crop_data["gray_0"] = np.zeros((1, *max_size), dtype=np.float32)
    crop_data["gray_1"] = np.zeros((1, *max_size), dtype=np.float32)
    crop_data["depth_0"] = np.zeros(max_size, dtype=np.int32)
    crop_data["depth_1"] = np.zeros(max_size, dtype=np.int32)

    gray0 = cv2.resize(gray0, (sizes[0,1], sizes[0,0]))
    gray1 = cv2.resize(gray1, (sizes[1,1], sizes[1,0]))
    depth0 = cv2.resize(depth0, (sizes[0,1], sizes[0,0]), interpolation=cv2.INTER_NEAREST)
    depth1 = cv2.resize(depth1, (sizes[1,1], sizes[1,0]), interpolation=cv2.INTER_NEAREST)
    
    crop_data["gray_0"][:, :sizes[0,0], :sizes[0,1]] = np.expand_dims(gray0, 0)
    crop_data["gray_1"][:, :sizes[1,0], :sizes[1,1]] = np.expand_dims(gray1, 0)
    crop_data["depth_0"][:sizes[0,0], :sizes[0,1]] = depth0
    crop_data["depth_1"][:sizes[1,0], :sizes[1,1]] = depth1

    crop_data['intrinsics_0'] = calculate_intrinsic_for_new_resolution(
        crop_data['intrinsics_0'], *tuple(sizes[0,[1,0]]), *tuple(old_sizes[0,[1,0]])
    )
    crop_data['intrinsics_1'] = calculate_intrinsic_for_new_resolution(
        crop_data['intrinsics_1'], *tuple(sizes[1,[1,0]]), *tuple(old_sizes[1,[1,0]])
    )

    if "seg_0" in crop_data:
        seg0 = crop_data["seg_0"].copy().astype(np.int16)
        seg1 = crop_data["seg_1"].copy().astype(np.int16)
        crop_data["seg_0"] = np.zeros(max_size, dtype=bool)
        crop_data["seg_1"] = np.zeros(max_size, dtype=bool)
        seg0 = cv2.resize(seg0, (sizes[0,1], sizes[0,0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        seg1 = cv2.resize(seg1, (sizes[1,1], sizes[1,0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        crop_data["seg_0"][:sizes[0,0], :sizes[0,1]] = seg0
        crop_data["seg_1"][:sizes[1,0], :sizes[1,1]] = seg1

    return crop_data

# def resize_to_common(crop_data):
#     gray0 = crop_data["gray_0"].copy()
#     gray1 = crop_data["gray_1"].copy()
#     depth0 = crop_data["depth_0"].copy()
#     depth1 = crop_data["depth_1"].copy()

#     sizes = np.vstack((
#         np.expand_dims(gray0.shape, 0),
#         np.expand_dims(gray1.shape, 0)
#         ))[:,1:]
#     max_dim_size = np.max(sizes)
#     # print(f"\n\n\n\n\n\n\n\n\n\n max size 1: {max_dim_size} ")
#     # print(max_dim_size / (16*8))
#     # print(max_dim_size % (16*8))
#     # max_dim_size = max_dim_size if max_dim_size % (16*8) == 0 else max_dim_size - (max_dim_size%(16*8))
#     # print(f"max size 2: {max_dim_size} ")
#     # max_dim_size = max_dim_size if max_dim_size > 256 else 256 
#     # max_size = tuple([max_dim_size]*2)
#     # print(f"MAX SIZE: {max_dim_size} \n\n\n\n\n\n\n\n\n\n")
#     # print(f"\n\n\n\n\n\n\n\n\n\n max size 1: {max_dim_size} ")
#     # print(max_dim_size / (16*8))
#     # print(max_dim_size % (16*8))
#     reduced_dim_in_model = (max_dim_size / 16)**2
#     reminder_after_attention = reduced_dim_in_model % 8
#     if reminder_after_attention == 0:
#         max_dim_size = max_dim_size 
#     else:
#         max_dim_size = int(16 *((max_dim_size / 16) + (4 - ((max_dim_size / 16) % 4))))
#         # max_dim_size = 16 * math.sqrt(reduced_dim_in_model + (8-reminder_after_attention))
#     # print(f"max size 2: {max_dim_size} ")
#     # print(f"max size 2 test: {(max_dim_size / 16)**2} ")
#     max_dim_size = max_dim_size if max_dim_size > 256 else 256 
#     max_size = tuple([max_dim_size]*2)
#     # print(f"MAX SIZE: {max_dim_size} \n\n\n\n\n\n\n\n\n\n")

#     crop_data["gray_0"] = np.zeros((1, *max_size), dtype=np.float32)
#     crop_data["gray_1"] = np.zeros((1, *max_size), dtype=np.float32)
#     crop_data["depth_0"] = np.zeros(max_size, dtype=np.int16)
#     crop_data["depth_1"] = np.zeros(max_size, dtype=np.int16)

#     crop_data["gray_0"][:, :sizes[0,0], :sizes[0,1]] = gray0
#     crop_data["gray_1"][:, :sizes[1,0], :sizes[1,1]] = gray1
#     crop_data["depth_0"][:sizes[0,0], :sizes[0,1]] = depth0
#     crop_data["depth_1"][:sizes[1,0], :sizes[1,1]] = depth1

#     return crop_data