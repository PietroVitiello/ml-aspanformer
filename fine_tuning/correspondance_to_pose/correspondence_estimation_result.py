import os.path
import cv2
import numpy as np
from copy import deepcopy

class CorrespondenceEstimationResult:
    """
    Class used to store correspondence estimation results
    """

    # def __init__(self, scene1_state: SceneState, scene2_state: SceneState, use_obj_crops=True):
    def __init__(self, use_obj_crops=True):
        # if use_obj_crops:
        #     assert scene1_state.obj_was_cropped, 'Object was not cropped for scene 1'
        #     assert scene2_state.obj_was_cropped, 'Object was not cropped for scene 2'

        self.use_obj_crops = use_obj_crops

        self._data: dict = None

        self._points1 = None
        self._points2 = None

        # self.scene1_state = deepcopy(scene1_state)
        # self.scene2_state = deepcopy(scene2_state)

        self._points1_inliers = None
        self._points2_inliers = None
        self._points1_outliers = None
        self._points2_outliers = None
        self._homography_matrix = None

    @property
    def data(self):
        return deepcopy(self._data)

    @data.setter
    def data(self, data: dict):
        self._data = data

    @property
    def points1(self):
        return self._points1.copy()

    @points1.setter
    def points1(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points1 = points

    @property
    def points2(self):
        return self._points2.copy()

    @points2.setter
    def points2(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 2 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 2 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points2 = points

    @property
    def outliers_removed(self):
        return (self._points1_inliers is not None) and (self._points2_inliers is not None)

    @property
    def points1_inliers(self):
        return self._points1_inliers.copy()

    @points1_inliers.setter
    def points1_inliers(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points1_inliers = points

    @property
    def points2_inliers(self):
        return self._points2_inliers.copy()

    @points2_inliers.setter
    def points2_inliers(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points2_inliers = points

    @property
    def points1_outliers(self):
        return self._points1_outliers.copy()

    @points1_outliers.setter
    def points1_outliers(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points1_outliers = points

    @property
    def points2_outliers(self):
        return self._points2_outliers.copy()

    @points2_outliers.setter
    def points2_outliers(self, points):
        assert isinstance(points, np.ndarray), 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert len(points.shape) == 2, 'Keypoints for scene 1 must be stored in a 2 dimensional numpy array'
        assert points.shape[1] == 2, 'The second dimension must store the (x, y) pixel location of each point'
        assert points.dtype == int, 'Each keypoint location must be an integer'

        self._points2_outliers = points

    @property
    def homography_matrix(self):
        return self._homography_matrix.copy()

    @homography_matrix.setter
    def homography_matrix(self, H):
        self._homography_matrix = H


class RansacOutlierRemoval:

    def __init__(self, method_type='USAC_DEFAULT', ransacReprojThreshold=3):

        assert method_type in ['USAC_DEFAULT', 'USAC_MAGSAC']
        if method_type == 'USAC_DEFAULT':
            self.ransac_method = cv2.USAC_DEFAULT
        elif method_type == 'USAC_MAGSAC':
            self.ransac_method == cv2.USAC_MAGSAC
        else:
            raise Exception(f'{method_type} is not a valid method type.')

        self.ransacReprojThreshold = ransacReprojThreshold

    @staticmethod
    def check_remove_outliers_input(correspondence_estimation_result: CorrespondenceEstimationResult):
        assert correspondence_estimation_result.points1 is not None
        assert correspondence_estimation_result.points2 is not None
        assert len(correspondence_estimation_result.points1) == len(
            correspondence_estimation_result.points1), 'The number of keypoints in the two images must be the same'
        assert len(
            correspondence_estimation_result.points1) > 3, 'More than 3 correspondences are required to solve for a pose'

    def remove_outliers(self,
                        correspondence_estimation_results: CorrespondenceEstimationResult,
                        visualise_warped_segmented_images=False,
                        visualise_inliers_and_outliers=False,
                        save_dir=None):

        RansacOutlierRemoval.check_remove_outliers_input(correspondence_estimation_results)

        points1 = correspondence_estimation_results.points1
        points2 = correspondence_estimation_results.points2

        src_pts = np.expand_dims(points1, axis=1)
        dst_pts = np.expand_dims(points2, axis=1)

        # compute homography matrix using RANSAC
        H, mask = cv2.findHomography(srcPoints=src_pts,
                                     dstPoints=dst_pts,
                                     method=self.ransac_method,
                                     ransacReprojThreshold=self.ransacReprojThreshold)

        # remove outliers
        correspondence_estimation_results.points1_inliers = points1[mask.ravel() == 1]
        correspondence_estimation_results.points2_inliers = points2[mask.ravel() == 1]

        correspondence_estimation_results.points1_outliers = points1[mask.ravel() == 0]
        correspondence_estimation_results.points2_outliers = points2[mask.ravel() == 0]

        correspondence_estimation_results.homography_matrix = H

        if visualise_warped_segmented_images:
            if save_dir is None:
                RansacOutlierRemoval.show_warped_segmented_images(
                    correspondence_estimation_results=correspondence_estimation_results,
                    mode='TkAgg',
                    save_path=None)
            else:
                save_path = os.path.join(save_dir, 'warped_segmented_images.png')
                RansacOutlierRemoval.show_warped_segmented_images(
                    correspondence_estimation_results=correspondence_estimation_results,
                    mode='Agg',
                    save_path=save_path)

        if visualise_inliers_and_outliers:
            if save_dir is None:
                RansacOutlierRemoval.show_inliers_and_outliers(
                    correspondence_estimation_results=correspondence_estimation_results,
                    mode='TkAgg',
                    save_path=None)
            else:
                save_path = os.path.join(save_dir, 'inliers_and_outliers.png')
                RansacOutlierRemoval.show_inliers_and_outliers(
                    correspondence_estimation_results=correspondence_estimation_results,
                    mode='Agg',
                    save_path=save_path)

        return correspondence_estimation_results
    
    # for visualisation functions check code in head_cam repo
