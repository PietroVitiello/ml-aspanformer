import matplotlib.pyplot as plt
import numpy as np
import cv2

from numba import njit, bool_
from .se3_tools import so3_log

def calculate_intrinsic_for_new_resolution(intrinsic: np.ndarray, new_width, new_height, old_width, old_height):
    # # See answer by Hammer: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    # n_width = np.log2(new_width / old_width)
    # n_height = np.log2(new_height / old_height)
    # transformation = np.array([[2 ** n_width, 0, 2 ** (n_width - 1) - 0.5],
    #                            [0, 2 ** n_height, 2 ** (n_height - 1) - 0.5],
    #                            [0, 0, 1]])
    # return transformation @ K
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height
    new_intrinsic = intrinsic.copy()
    new_intrinsic[0] *= ratio_width
    new_intrinsic[1] *= ratio_height
    return new_intrinsic

def calculate_rot_delta(R):
    rotvec = so3_log(R)
    return np.rad2deg(np.linalg.norm(rotvec))

@njit
def rgb2gray(rgb):
    # Compiling with numba does not penalise for loops and only supports 2D arrays so embrace the for loop and go on
    gray = np.zeros((rgb.shape[0], rgb.shape[1], 1))
    scaling = np.ascontiguousarray(np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    for channel in range(rgb.shape[2]):
        gray[:,:,0] += rgb[:,:, channel] * scaling[channel]
    return gray

@njit
def pose_inv(pose):
    """
    Function used to convert a pose to an extrinsic matrix.
    Args:
        posevec: Position concatenated with a quaternion.
    Returns:
        T_OW: Extrinsic Matrix. This is a transformation that can be used to map a point in the world frame to the
            object frame.
    """
    R = pose[:3, :3]

    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = - R.T @ np.ascontiguousarray(pose[:3, 3])

    return T

@njit
def calculate_intrinsic_for_crop(intrinsic: np.ndarray, top: int, left: int):
    x0, y0 = intrinsic[0:2, 2]
    x0 = x0 - left
    y0 = y0 - top
    intrinsic[0:2, 2] = [x0, y0]
    return intrinsic

@njit
def calculate_intrinsic_for_new_resolution(intrinsic_matrix: np.ndarray, new_width, new_height, old_width, old_height):
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height
    new_intrinsic = intrinsic_matrix.copy()
    new_intrinsic[0] *= ratio_width
    new_intrinsic[1] *= ratio_height
    return new_intrinsic

@njit
def bbox_from_mask(mask, margin: float = 0.2):
    y, x = np.where(mask != 0)

    bbox = np.array([
        np.min(x),
        np.min(y),
        np.max(x),
        np.max(y)
    ])

    if margin == 0:
        return bbox

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    margin = np.array([-width, -height, width, height]) * margin / 2
    return bbox + margin.astype(np.int16)

@njit
def crop(bbox, rgb, depth, segmap):
    xmin, ymin, xmax, ymax = bbox
    xmin = xmin if xmin > 0 else 0
    ymin = ymin if ymin > 0 else 0
    xmax = xmax if xmax < rgb.shape[1] else rgb.shape[1]
    ymax = ymax if ymax < rgb.shape[0] else rgb.shape[0]
    
    rgb = rgb[ymin:ymax, xmin:xmax]
    depth = depth[ymin:ymax, xmin:xmax]
    segmap = segmap[ymin:ymax, xmin:xmax]

    bbox = (xmin, ymin, xmax, ymax)
    return rgb, depth, segmap, bbox

# def crop(bbox, *imgs, **kwargs):
#     if "return_updated_bbox" not in kwargs.keys():
#         kwargs["return_updated_bbox"] = False
#     xmin, ymin, xmax, ymax = bbox
#     xmin = xmin if xmin > 0 else 0
#     ymin = ymin if ymin > 0 else 0
#     xmax = xmax if xmax < imgs[0].shape[1] else imgs[0].shape[1]
#     ymax = ymax if ymax < imgs[0].shape[0] else imgs[0].shape[0]
    
#     cropped_imgs = []
#     for i, img in enumerate(imgs):
#         cropped_imgs.append(img[ymin:ymax, xmin:xmax])

#     if kwargs["return_updated_bbox"]:
#         bbox = (xmin, ymin, xmax, ymax)
#         return *cropped_imgs, bbox
#     return cropped_imgs


@njit
def unique_counts(ar: np.ndarray):
    """
    Implementation of np.unique that allows for compilation

    Args:
     - ar: np.array [N,2]
    """

    unique = ar.copy()
    counts = np.ones(unique.shape[0])
    for i in range(ar.shape[0]):
        if counts[i] > 0:
            for other_i in range(ar.shape[0]):
                if i != other_i:
                    if (ar[i,:] == ar[other_i,:]).all():
                        counts[i] += 1
                        counts[other_i] -= 1
    valid = counts > 0
    unique = unique[valid,:]
    counts = counts[valid]
    return unique, counts

def get_keypoint_indices(segmap, coarse_factor=1):
    x = np.arange(segmap.shape[1], step=coarse_factor, dtype=int)  # TODO should we have '+ 0.5' here?
    y = np.arange(segmap.shape[0], step=coarse_factor, dtype=int)  # TODO should we have '+ 0.5' here?
    xx, yy = np.meshgrid(x, y)
    indices = np.concatenate((xx[..., None], yy[..., None]), axis=2)
    return indices[segmap[y,:][:,x]]


@njit
def estimate_cropped_correspondences(keypoints: np.ndarray,
                                     depth_image_0: np.ndarray,
                                     depth_image_1: np.ndarray,
                                     T_01: np.ndarray,
                                     K0: np.ndarray,
                                     K1: np.ndarray,
                                     depth_units: str = 'mm',
                                     depth_rejection_threshold: float = 0.001):

        # Get the depth value of the keypoint
        depth_values_0 = np.empty((keypoints.shape[0], 1))
        for i in range(keypoints.shape[0]):
            depth_values_0[i,0] = depth_image_0[keypoints[i, 1], keypoints[i, 0]]

        if depth_units == 'mm':
            depth_values_0 = depth_values_0 / 1e3

        # Get the position of the keypoint in the camera frame
        keypoint_pos_C_0 = depth_values_0 * np.concatenate((keypoints, np.ones((len(keypoints), 1))),
                                                        axis=1) @ np.linalg.inv(K0).T

        # Calculate the coordinates of the new point in the second image.
        keypoint_pos_C_1 = np.concatenate((keypoint_pos_C_0, np.ones((len(keypoint_pos_C_0), 1))), axis=1) @ pose_inv(T_01).T

        # Use this depth value to determine if a match was found
        expected_depth_values_1 = keypoint_pos_C_1[:, 2:3].copy()

        # Project the point to image plane
        keypoint_pos_C_1 = keypoint_pos_C_1[:, :3]
        keypoint_pos_C_1 /= keypoint_pos_C_1[:, 2:3]

        keypoints_1 = (np.ascontiguousarray(keypoint_pos_C_1) @ np.ascontiguousarray(K1.T))[:, :2]
        keypoints_1_rounded = np.round(keypoints_1, 0, keypoints_1)
        keypoints_1_rounded = keypoints_1_rounded.astype(np.int64)
        residuals = keypoints_1 - keypoints_1_rounded
        residuals = np.sqrt((residuals * residuals).sum(axis=1))

        # Check which correspondences are valid
        valid = np.ones(len(keypoints), dtype=bool_)
        # valid = np.ones(len(keypoints), dtype=bool)

        # Reject all correspondences outside the image plane
        valid[keypoints_1[:, 0] < 0] = False
        valid[keypoints_1[:, 1] < 0] = False
        valid[keypoints_1[:, 0] > depth_image_1.shape[1] - 1] = False
        valid[keypoints_1[:, 1] > depth_image_1.shape[0] - 1] = False

        # Check if expected depth values match actual depth values
        keypoints_rounded_valid = keypoints_1_rounded[valid]
        depth_values_1 = np.zeros((keypoints_rounded_valid.shape[0], 1))
        for i in range(keypoints_rounded_valid.shape[0]):
            depth_values_1[i,0] = depth_image_1[keypoints_rounded_valid[i, 1], keypoints_rounded_valid[i, 0]]
        
        if depth_units == 'mm':
            depth_values_1 = depth_values_1 / 1e3

        delta_depth_values = np.abs(depth_values_1 - expected_depth_values_1[valid])

        # Hacky way to set valid flag to false for all points for which there is a change in depth larger than a threshold
        indices_of_not_valid = np.arange(len(keypoints))
        indices_of_not_valid = indices_of_not_valid[valid]
        indices_of_not_valid = indices_of_not_valid[(delta_depth_values > depth_rejection_threshold).flatten()]
        valid[indices_of_not_valid] = False

        keypoints_1_rounded[np.logical_not(valid)] = np.array([-1, -1])

        indices_of_valid = np.arange(len(keypoints))
        valid_keypoints_1 = keypoints_1_rounded[valid]
        # print(f"valid kpts: {valid_keypoints_1.shape}")
        # unique, counts = np.unique(valid_keypoints_1, axis=0, return_counts=True)
        # print(f"unique: {unique.shape}")
        unique, counts = unique_counts(valid_keypoints_1)

        if (counts > 1).any():
            for kpt in unique[counts > 1]:
                equal_keypoints = np.zeros(indices_of_valid.shape[0], dtype=bool_)
                # equal_keypoints = np.zeros(indices_of_valid.shape[0], dtype=bool)
                for i in range(indices_of_valid.shape[0]):
                    equal_keypoints[i] = (keypoints_1_rounded[i,:] == kpt).all()
                kpt_indices = indices_of_valid[equal_keypoints]
                idx_of_closest_kpt = kpt_indices[np.argmin(residuals[kpt_indices])]
                for idx in kpt_indices:
                    if idx != idx_of_closest_kpt:
                        keypoints_1_rounded[idx,:] = np.array([-1, -1])
                # keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = np.array([-1, -1])
                for idx in kpt_indices:
                    if idx != idx_of_closest_kpt:
                        valid[idx] = False
                # valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False

        return np.concatenate((keypoints_1_rounded, np.expand_dims(valid, axis=-1)), axis=1)

        # if return_valid_list:
        #     return keypoints_1_rounded, valid
        # else:
        #     return keypoints_1_rounded
    

def plot_matches(rgb_0: np.ndarray, kpts_0, rgb_1: np.ndarray, kpts_1, num_points_to_plot=-1, shuffle_matches=False,
                 match_flag=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS):
    
    rgb_0 = (rgb_0 * 255).transpose(1,2,0).astype(np.uint8)
    rgb_1 = (rgb_1 * 255).transpose(1,2,0).astype(np.uint8)
    size = 1
    distance = 1
    # Create keypoints
    keypoints_0_cv = []
    keypoints_1_cv = []
    for kpt_0, kpt_1 in zip(kpts_0, kpts_1):
        if not (kpt_1 == np.array([-1, -1])).all():
            keypoints_0_cv.append(cv2.KeyPoint(x=float(kpt_0[0]), y=float(kpt_0[1]), size=size))
            keypoints_1_cv.append(cv2.KeyPoint(x=float(kpt_1[0]), y=float(kpt_1[1]), size=size))
    keypoints_0_cv = tuple(keypoints_0_cv)
    keypoints_1_cv = tuple(keypoints_1_cv)

    # Create a list of matches
    matches = []
    for idx in range(len(keypoints_0_cv)):
        match = cv2.DMatch()
        match.trainIdx = idx
        match.queryIdx = idx
        match.trainIdx = idx
        match.distance = distance
        matches.append(match)

    if shuffle_matches:
        # Shuffle all matches
        matches = list(np.array(matches)[np.random.permutation(len(matches))])

    img = cv2.drawMatches(rgb_0, keypoints_0_cv, rgb_1, keypoints_1_cv, matches[:num_points_to_plot], None,
                          flags=match_flag)
    
    plt.figure()
    plt.imshow(rgb_0)
    plt.figure()
    plt.imshow(rgb_1)

    plt.figure()
    plt.imshow(img)
    plt.show()
    return img
