from .utils import pose_inv
import numpy as np

def unique_counts(ar: np.ndarray):
    """
    Implementation of np.unique that allows for compilation

    Args:
     - ar: np.array [N,2]
    """

    unique = ar.copy()
    counts = np.ones(unique.shape[0])
    for i in range(ar.shape[0]):
        # other_kpts = np.ones((valid_keypoints_1.shape[0]-1, valid_keypoints_1.shape[1]))
        # other_kpts[0:i, :] = valid_keypoints_1[0:i, :]
        # other_kpts[i:, :] = valid_keypoints_1[i+1:, :]
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
        
def estimate_cropped_correspondences(keypoints: np.ndarray,
                                     depth_image_0: np.ndarray,
                                     depth_image_1: np.ndarray,
                                     T_01: np.ndarray,
                                     K0: np.ndarray,
                                     K1: np.ndarray,
                                     depth_units: str = 'mm',
                                     depth_rejection_threshold: float = 0.001,
                                     return_valid_list: bool = True):

        assert isinstance(keypoints, np.ndarray), 'Keypoints must be stored in a numpy array'
        assert keypoints.dtype == np.int64, 'Keypoints should be integers'
        assert len(keypoints.shape) == 2, 'Keypoints must be stored in a 2-dimensional array'
        assert keypoints.shape[1] == 2, 'The x and y position of all keypoints must be specified'
        assert depth_units in ['m', 'mm'], 'Depth units must be either meters or millimeters'

        # Get the depth value of the keypoint
        depth_values_0 = depth_image_0[keypoints[:, 1], keypoints[:, 0]].reshape(-1, 1)

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

        keypoints_1 = (keypoint_pos_C_1 @ K1.T)[:, :2]
        keypoints_1_rounded = np.round(keypoints_1).astype(np.int64)
        residuals = keypoints_1 - keypoints_1_rounded
        residuals = np.sqrt((residuals * residuals).sum(axis=1))

        # Check which correspondences are valid
        valid = np.ones(len(keypoints), dtype=bool)

        # Reject all correspondences outside the image plane
        valid[keypoints_1[:, 0] < 0] = False
        valid[keypoints_1[:, 1] < 0] = False
        valid[keypoints_1[:, 0] > depth_image_1.shape[1] - 1] = False
        valid[keypoints_1[:, 1] > depth_image_1.shape[0] - 1] = False

        # Check if expected depth values match actual depth values

        depth_values_1 = depth_image_1[keypoints_1_rounded[valid][:, 1], keypoints_1_rounded[valid][:, 0]].reshape(-1, 1)
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
        # unique, counts = np.unique(valid_keypoints_1, axis=0, return_counts=True)
        unique, counts = unique_counts(valid_keypoints_1)

        if (counts > 1).any():
            for kpt in unique[counts > 1]:
                kpt_indices = indices_of_valid[(keypoints_1_rounded == kpt).all(axis=1)]
                idx_of_closest_kpt = kpt_indices[np.argmin(residuals[kpt_indices])]
                keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = np.array([-1, -1])
                valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False

        if return_valid_list:
            return np.concatenate((keypoints_1_rounded, valid[:,None]), axis=1)
        else:
            return keypoints_1_rounded

def estimate_correspondences_diff_intr(keypoints: np.ndarray,
                                       depth_image_0: np.ndarray,
                                       depth_image_1: np.ndarray,
                                       T_CO_0: np.ndarray,
                                       T_CO_1: np.ndarray,
                                       K0: np.ndarray,
                                       K1: np.ndarray,
                                       depth_units: str = 'mm',
                                       depth_rejection_threshold: float = 0.001,
                                       return_valid_list: bool = False):

        assert isinstance(keypoints, np.ndarray), 'Keypoints must be stored in a numpy array'
        assert keypoints.dtype == np.int64, 'Keypoints should be integers'
        assert len(keypoints.shape) == 2, 'Keypoints must be stored in a 2-dimensional array'
        assert keypoints.shape[1] == 2, 'The x and y position of all keypoints must be specified'
        assert depth_units in ['m', 'mm'], 'Depth units must be either meters or millimeters'

        # Get the depth value of the keypoint
        depth_values_0 = depth_image_0[keypoints[:, 1], keypoints[:, 0]].reshape(-1, 1)

        if depth_units == 'mm':
            depth_values_0 = depth_values_0 / 1e3

        # Get the position of the keypoint in the camera frame
        keypoint_pos_C_0 = depth_values_0 * np.concatenate((keypoints, np.ones((len(keypoints), 1))),
                                                        axis=1) @ np.linalg.inv(K0).T

        T_delta = T_CO_1 @ pose_inv(T_CO_0)

        # Calculate the coordinates of the new point in the second image.
        keypoint_pos_C_1 = np.concatenate((keypoint_pos_C_0, np.ones((len(keypoint_pos_C_0), 1))), axis=1) @ T_delta.T

        # Use this depth value to determine if a match was found
        expected_depth_values_1 = keypoint_pos_C_1[:, 2:3].copy()

        # Project the point to image plane
        keypoint_pos_C_1 = keypoint_pos_C_1[:, :3]
        keypoint_pos_C_1 /= keypoint_pos_C_1[:, 2:3]

        keypoints_1 = (keypoint_pos_C_1 @ K1.T)[:, :2]
        keypoints_1_rounded = np.round(keypoints_1).astype(np.int64)
        residuals = keypoints_1 - keypoints_1_rounded
        residuals = np.sqrt((residuals * residuals).sum(axis=1))

        # Check which correspondences are valid
        valid = np.ones(len(keypoints), dtype=bool)

        # Reject all correspondences outside the image plane
        valid[keypoints_1[:, 0] < 0] = False
        valid[keypoints_1[:, 1] < 0] = False
        valid[keypoints_1[:, 0] > depth_image_1.shape[1] - 1] = False
        valid[keypoints_1[:, 1] > depth_image_1.shape[0] - 1] = False

        # Check if expected depth values match actual depth values
        depth_values_1 = depth_image_1[keypoints_1_rounded[valid][:, 1], keypoints_1_rounded[valid][:, 0]].reshape(-1, 1)
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
        unique, counts = np.unique(valid_keypoints_1, axis=0, return_counts=True)

        if (counts > 1).any():
            for kpt in unique[counts > 1]:
                kpt_indices = indices_of_valid[(keypoints_1_rounded == kpt).all(axis=1)]
                idx_of_closest_kpt = kpt_indices[np.argmin(residuals[kpt_indices])]
                keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = np.array([-1, -1])
                valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False

        if return_valid_list:
            return keypoints_1_rounded, valid
        else:
            return keypoints_1_rounded

def estimate_correspondences(keypoints: np.ndarray,
                             depth_image_0: np.ndarray,
                             depth_image_1: np.ndarray,
                             T_CO_0: np.ndarray,
                             T_CO_1: np.ndarray,
                             intrinsics: np.ndarray,
                             depth_units: str = 'mm',
                             depth_rejection_threshold: float = 0.001,
                             return_valid_list: bool = False):

        assert isinstance(keypoints, np.ndarray), 'Keypoints must be stored in a numpy array'
        assert keypoints.dtype == np.int64, 'Keypoints should be integers'
        assert len(keypoints.shape) == 2, 'Keypoints must be stored in a 2-dimensional array'
        assert keypoints.shape[1] == 2, 'The x and y position of all keypoints must be specified'
        assert depth_units in ['m', 'mm'], 'Depth units must be either meters or millimeters'

        # Get the depth value of the keypoint
        depth_values_0 = depth_image_0[keypoints[:, 1], keypoints[:, 0]].reshape(-1, 1)

        if depth_units == 'mm':
            depth_values_0 = depth_values_0 / 1e3

        # Get the position of the keypoint in the camera frame
        keypoint_pos_C_0 = depth_values_0 * np.concatenate((keypoints, np.ones((len(keypoints), 1))),
                                                        axis=1) @ np.linalg.inv(intrinsics).T

        # Calculate the movement of the object between the two images, i.e. T_CO_1 = T_delta @ T_CO_0
        T_delta = T_CO_1 @ pose_inv(T_CO_0)

        # Calculate the coordinates of the new point in the second image.
        keypoint_pos_C_1 = np.concatenate((keypoint_pos_C_0, np.ones((len(keypoint_pos_C_0), 1))), axis=1) @ T_delta.T

        # Use this depth value to determine if a match was found
        expected_depth_values_1 = keypoint_pos_C_1[:, 2:3].copy()

        # Project the point to image plane
        keypoint_pos_C_1 = keypoint_pos_C_1[:, :3]
        keypoint_pos_C_1 /= keypoint_pos_C_1[:, 2:3]

        keypoints_1 = (keypoint_pos_C_1 @ intrinsics.T)[:, :2]
        keypoints_1_rounded = np.round(keypoints_1).astype(np.int64)
        residuals = keypoints_1 - keypoints_1_rounded
        residuals = np.sqrt((residuals * residuals).sum(axis=1))

        # Check which correspondences are valid
        valid = np.ones(len(keypoints), dtype=bool)

        # Reject all correspondences outside the image plane
        valid[keypoints_1[:, 0] < 0] = False
        valid[keypoints_1[:, 1] < 0] = False
        valid[keypoints_1[:, 0] > depth_image_0.shape[1] - 1] = False
        valid[keypoints_1[:, 1] > depth_image_0.shape[0] - 1] = False

        # Check if expected depth values match actual depth values

        depth_values_1 = depth_image_1[keypoints_1_rounded[valid][:, 1], keypoints_1_rounded[valid][:, 0]].reshape(-1, 1)
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
        unique, counts = np.unique(valid_keypoints_1, axis=0, return_counts=True)

        if (counts > 1).any():
            for kpt in unique[counts > 1]:
                kpt_indices = indices_of_valid[(keypoints_1_rounded == kpt).all(axis=1)]
                idx_of_closest_kpt = kpt_indices[np.argmin(residuals[kpt_indices])]
                keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = np.array([-1, -1])
                valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False

        if return_valid_list:
            return keypoints_1_rounded, valid
        else:
            return keypoints_1_rounded