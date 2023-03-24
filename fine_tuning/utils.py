import matplotlib.pyplot as plt
import numpy as np
import cv2

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return np.expand_dims(gray, axis=-1)

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

def calculate_intrinsic_for_crop(intrinsic: np.ndarray, top: int, left: int):
    x0, y0 = intrinsic[0:2, 2]
    x0 = x0 - left
    y0 = y0 - top
    intrinsic[0:2, 2] = [x0, y0]
    return intrinsic

def calculate_intrinsic_for_new_resolution(intrinsic_matrix: np.ndarray, new_width, new_height, old_width, old_height):
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height
    new_intrinsic = intrinsic_matrix.copy()
    new_intrinsic[0] *= ratio_width
    new_intrinsic[1] *= ratio_height
    return new_intrinsic

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

def crop(bbox, *imgs, **kwargs):
    if "return_updated_bbox" not in kwargs.keys():
        kwargs["return_updated_bbox"] = False
    xmin, ymin, xmax, ymax = bbox
    xmin = xmin if xmin > 0 else 0
    ymin = ymin if ymin > 0 else 0
    xmax = xmax if xmax < imgs[0].shape[1] else imgs[0].shape[1]
    ymax = ymax if ymax < imgs[0].shape[0] else imgs[0].shape[0]
    
    cropped_imgs = []
    for i, img in enumerate(imgs):
        cropped_imgs.append(img[ymin:ymax, xmin:xmax])

    if kwargs["return_updated_bbox"]:
        bbox = (xmin, ymin, xmax, ymax)
        return *cropped_imgs, bbox
    return cropped_imgs

def get_keypoint_indices(segmap):
    x = np.arange(segmap.shape[1])  # TODO should we have '+ 0.5' here?
    y = np.arange(segmap.shape[0])  # TODO should we have '+ 0.5' here?
    xx, yy = np.meshgrid(x, y)
    indices = np.concatenate((xx[..., None], yy[..., None]), axis=2)
    return indices[segmap]

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
