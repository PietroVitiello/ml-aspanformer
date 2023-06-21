from abc import ABC, abstractmethod

import numpy as np

from fine_tuning.correspondance_to_pose.correspondence_estimation_result import CorrespondenceEstimationResult


class Correspondence2PoseBase(ABC):

    def __init__(self, depth_units='mm', **kwargs):
        assert depth_units in ['mm', 'm'], 'Depth units must be \'mm\' (millimeters) or \'m\' (meters)'
        self.depth_units = depth_units

    @abstractmethod
    def estimate_relative_pose(self,
                               correspondence_estimation_result: CorrespondenceEstimationResult,
                               **kwargs) -> np.ndarray:
        """
        :return: Relative transformation calculated using the correspondences
        """
        pass

    @staticmethod
    def validate_estimate_relative_pose_inputs(correspondence_estimation_result: CorrespondenceEstimationResult):
        assert correspondence_estimation_result.points1 is not None
        assert correspondence_estimation_result.points2 is not None
        assert len(correspondence_estimation_result.points1) == len(correspondence_estimation_result.points1), 'The number of keypoints in the two images must be the same'
        assert len(correspondence_estimation_result.points1) > 3, 'More than 3 correspondences are required to solve for a pose'
