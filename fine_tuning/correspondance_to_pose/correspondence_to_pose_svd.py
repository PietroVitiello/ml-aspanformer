import numpy as np
import torch
from torch import nn


from fine_tuning.correspondance_to_pose.correspondence_estimation_result import CorrespondenceEstimationResult
from fine_tuning.correspondance_to_pose.correspondence_estimation_result import RansacOutlierRemoval
from fine_tuning.correspondance_to_pose.correspondence_to_pose import Correspondence2PoseBase
from fine_tuning.correspondance_to_pose.img_to_pcd import get_3d_points_from_pixel_indices


def create_corrEst_result(data: dict) -> CorrespondenceEstimationResult:
    res_container = CorrespondenceEstimationResult(use_obj_crops=True)
    res_container.data = data
    return res_container

def get_predicted_T(data: dict) -> np.ndarray:
    svd = CorrespondenceToPoseSVD()
    return svd.estimate_relative_pose(data)

class SVDHead(nn.Module):
    def __init__(self, emb_dims=32):
        super(SVDHead, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, tgt):
        batch_size = src.size(0)

        src_centered = src - src.mean(dim=2, keepdim=True)
        tgt_centered = tgt - tgt.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, tgt_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)


class CorrespondenceToPoseSVD(Correspondence2PoseBase):

    def __init__(self,
                 depth_units='mm',
                 dtype: torch.dtype = torch.float32):
        self.svd_head = SVDHead()
        self.dtype = dtype

        assert depth_units == 'mm'

        super().__init__(depth_units=depth_units)
        self.ransac_outlier_removal = RansacOutlierRemoval(method_type='USAC_DEFAULT',
                                                           ransacReprojThreshold=5)

    def estimate_relative_pose(self, data: dict) -> np.ndarray:

        # Validate all inputs
        T_preds = np.zeros((data["depth0"].shape[0],4,4))
        correspondence_estimation_result = create_corrEst_result(data)

        for i in range(data["depth0"].shape[0]):
            correspondence_estimation_result.points1 = data['mkpts0_f'][data["m_bids"]==i].cpu().numpy().astype(int)
            correspondence_estimation_result.points2 = data['mkpts1_f'][data["m_bids"]==i].cpu().numpy().astype(int)    
            self.ransac_outlier_removal.remove_outliers(correspondence_estimation_result)
            CorrespondenceToPoseSVD.validate_estimate_relative_pose_inputs(correspondence_estimation_result)

            points1 = correspondence_estimation_result.points1_inliers
            points2 = correspondence_estimation_result.points2_inliers
            # print(correspondence_estimation_result.data.keys())
            depth1 = correspondence_estimation_result.data["depth0"][i].cpu().numpy()
            depth2 = correspondence_estimation_result.data["depth1"][i].cpu().numpy()

            intrinsic_matrix_1 = correspondence_estimation_result.data["K0"][i].cpu().numpy()
            intrinsic_matrix_2 = correspondence_estimation_result.data["K1"][i].cpu().numpy()

            # Get the xyz position of every point in both images
            p_frame_1 = get_3d_points_from_pixel_indices(pixel_indices=points1, intrinsic_matrix=intrinsic_matrix_1,
                                                        depth_map=depth1, depth_units=self.depth_units)
            p_frame_2 = get_3d_points_from_pixel_indices(pixel_indices=points2, intrinsic_matrix=intrinsic_matrix_2,
                                                        depth_map=depth2, depth_units=self.depth_units)

            # Reshape to (batch_size, 3, num_points) for compatibility with the SVD head
            p_frame_1 = torch.tensor(p_frame_1, dtype=self.dtype).permute(1, 0).unsqueeze(0)
            p_frame_2 = torch.tensor(p_frame_2, dtype=self.dtype).permute(1, 0).unsqueeze(0)

            # Estimate the pose
            R_pred, t_pred = self.svd_head(p_frame_1, p_frame_2)

            T_pred = np.eye(4)
            T_pred[:3, :3] = R_pred[0].cpu().numpy()
            T_pred[:3, 3] = t_pred[0].cpu().numpy()
            T_preds[i,:,:] = T_pred

        return T_preds
