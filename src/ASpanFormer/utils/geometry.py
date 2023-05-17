import torch




# import matplotlib.pyplot as plt
# import cv2
# import numpy as np


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, img0, img1):
    #TODO: Remember to change the arguments needed when removing the plot
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long() # Batch, u, v

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    # NOTE: Added depth consistency. If using Scannet or MegaDepth revert to original code below:
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    consistent_mask = (w_kpts0_depth - w_kpts0_depth_computed).abs() < 0.003 # depth consistency
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    w_kpts0[~valid_mask, :] = 0
    # print(f"valid: {(valid_mask[valid_mask == 1]).shape} \n")

    # print(w_kpts0.shape)
    # print(w_kpts0[valid_mask].shape)
    # print(kpts0.shape)


    # rgb_0 = (img0.cpu().numpy() * 255).transpose(1,2,0).astype(np.uint8)
    # rgb_1 = (img1.cpu().numpy() * 255).transpose(1,2,0).astype(np.uint8)
    # size = 1
    # distance = 1
    # # Create keypoints
    # keypoints_0_cv = []
    # keypoints_1_cv = []
    # # print(w_kpts0[valid_mask][0].shape)
    # for kpt_0, kpt_1 in zip(kpts0[valid_mask].cpu().numpy(), w_kpts0[valid_mask].cpu().numpy()):
    #     # print(kpt_0.shape, kpt_1.shape)
    #     if not (kpt_1 == np.array([-1, -1])).all():
    #         keypoints_0_cv.append(cv2.KeyPoint(x=float(kpt_0[0]), y=float(kpt_0[1]), size=size))
    #         keypoints_1_cv.append(cv2.KeyPoint(x=float(kpt_1[0]), y=float(kpt_1[1]), size=size))
    # keypoints_0_cv = tuple(keypoints_0_cv)
    # keypoints_1_cv = tuple(keypoints_1_cv)

    # # Create a list of matches
    # matches = []
    # for idx in range(len(keypoints_0_cv)):
    #     match = cv2.DMatch()
    #     match.trainIdx = idx
    #     match.queryIdx = idx
    #     match.trainIdx = idx
    #     match.distance = distance
    #     matches.append(match)

    # img = cv2.drawMatches(rgb_0, keypoints_0_cv, rgb_1, keypoints_1_cv, matches[:-1], None,
    #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # plt.figure()
    # plt.imshow(rgb_0)
    # plt.figure()
    # plt.imshow(rgb_1)

    # for f in plt.get_fignums():
    #     plt.close(f)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    return valid_mask, w_kpts0

########################### Original
# @torch.no_grad()
# def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
#     """ Warp kpts0 from I0 to I1 with depth, K and Rt
#     Also check covisibility and depth consistency.
#     Depth is consistent if relative error < 0.2 (hard-coded).
    
#     Args:
#         kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
#         depth0 (torch.Tensor): [N, H, W],
#         depth1 (torch.Tensor): [N, H, W],
#         T_0to1 (torch.Tensor): [N, 3, 4],
#         K0 (torch.Tensor): [N, 3, 3],
#         K1 (torch.Tensor): [N, 3, 3],
#     Returns:
#         calculable_mask (torch.Tensor): [N, L]
#         warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
#     """
#     kpts0_long = kpts0.round().long() # Batch, u, v

#     # Sample depth, get calculable_mask on depth != 0
#     kpts0_depth = torch.stack(
#         [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
#     )  # (N, L)
#     nonzero_mask = kpts0_depth != 0

#     # Unproject
#     kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
#     kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

#     # Rigid Transform
#     w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
#     w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

#     # Project
#     w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
#     w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

#     # Covisible Check
#     h, w = depth1.shape[1:3]
#     covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
#         (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
#     w_kpts0_long = w_kpts0.long()
#     w_kpts0_long[~covisible_mask, :] = 0
#     # print(f"kpts: {w_kpts0_long.shape}")
#     # print(f"covisible: {w_kpts0_long[w_kpts0_long != 0].shape}")


#     w_kpts0_depth = torch.stack(
#         [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
#     )  # (N, L)
#     # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.02 #0.2
#     # consistent_mask = (w_kpts0_depth - w_kpts0_depth_computed).abs() > 100
#     # print("\n\n\n\n\n\n\n\n\n")
#     # print(f"really??: {consistent_mask[consistent_mask != 0].shape}")
#     consistent_mask = (w_kpts0_depth - w_kpts0_depth_computed).abs() < 0.001
#     # print(f"consistent: {(consistent_mask[consistent_mask == 1]).shape}")
#     valid_mask = nonzero_mask * covisible_mask * consistent_mask
#     w_kpts0[~valid_mask, :] = 0
#     # print(f"valid: {(valid_mask[valid_mask == 1]).shape} \n")

#     return valid_mask, w_kpts0















# @torch.no_grad()
# def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
#     """ Warp kpts0 from I0 to I1 with depth, K and Rt
#     Also check covisibility and depth consistency.
#     Depth is consistent if relative error < 0.2 (hard-coded).
    
#     Args:
#         kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
#         depth0 (torch.Tensor): [N, H, W],
#         depth1 (torch.Tensor): [N, H, W],
#         T_0to1 (torch.Tensor): [N, 3, 4],
#         K0 (torch.Tensor): [N, 3, 3],
#         K1 (torch.Tensor): [N, 3, 3],
#     Returns:
#         calculable_mask (torch.Tensor): [N, L]
#         warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
#     """
    ####################################################
    # kpts0_long = kpts0.round().long() # Batch, u, v

    # # Sample depth, get calculable_mask on depth != 0
    # kpts0_depth = torch.stack(
    #     [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    # )  # (N, L)
    # nonzero_mask = kpts0_depth != 0

    # # Unproject
    # kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    # kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # # Rigid Transform
    # w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    # w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # # Project
    # w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    # w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # # Covisible Check
    # h, w = depth1.shape[1:3]
    # covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0
    # # print(f"kpts: {w_kpts0_long.shape}")
    # # print(f"covisible: {w_kpts0_long[w_kpts0_long != 0].shape}")


    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.02 #0.2
    # consistent_mask = (w_kpts0_depth - w_kpts0_depth_computed).abs() < 0.02
    # # print(f"consistent: {(consistent_mask[consistent_mask == 1]).shape}")
    # valid_mask = nonzero_mask * covisible_mask * consistent_mask
    # # print(f"valid: {(valid_mask[valid_mask == 1]).shape} ")
    # print(f"valid: {(valid_mask).shape} ")
    # print(f"valid: {(w_kpts0).shape} ")

    # return valid_mask, w_kpts0




    
    # # Get the depth value of the keypoint
    # kpts0_long = kpts0.round().long() # Batch, u, v
    # bs = kpts0.shape[0]
    # depth_values_0 = torch.stack(
    #     [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    # )  # (N, L)
    # nonzero_mask = depth_values_0 != 0

    # # Get the position of the keypoint in the camera frame
    # print(f"Just curious: {kpts0[:, :, [0]].shape}")
    # print(f"Depth: {depth_values_0.shape}")
    # print(f"K: {K0.inverse().shape}")
    # # print(f"kpts: {depth_values_0 * torch.cat((kpts0, torch.ones_like(kpts0[:, :, [0]])), dim=-1).shape}")
    # print((torch.cat((kpts0, torch.ones_like(kpts0[:, :, [0]])), dim=-1) @ K0.inverse().transpose(2,1)).shape)
    # keypoint_pos_C_0 = depth_values_0[...,None] * torch.cat((kpts0, torch.ones_like(kpts0[:, :, [0]])), dim=-1) @ K0.inverse().transpose(2,1)

    # print(f"T: {T_0to1.transpose(2,1).shape}")
    # print(f"kpts: {torch.cat((keypoint_pos_C_0, torch.ones_like((kpts0[:, :, [0]]))), dim=-1).shape}")
    # # Calculate the coordinates of the new point in the second image.
    # keypoint_pos_C_1 = torch.cat((keypoint_pos_C_0, torch.ones_like((kpts0[:, :, [0]]))), dim=-1) @ T_0to1.transpose(2,1)

    # # Use this depth value to determine if a match was found
    # expected_depth_values_1 = keypoint_pos_C_1[:, :, 2:3].detach().clone()

    # # Project the point to image plane
    # keypoint_pos_C_1 = keypoint_pos_C_1[:, :, :3]
    # keypoint_pos_C_1 /= keypoint_pos_C_1[:, :, 2:3]

    # keypoints_1 = (keypoint_pos_C_1 @ K1.T)[:, :, :2]
    # keypoints_1_rounded = torch.round(keypoints_1).to(dtype=torch.int64)
    # residuals = keypoints_1 - keypoints_1_rounded
    # residuals = torch.sqrt((residuals * residuals).sum(axis=1))

    # # Check which correspondences are valid
    # valid = torch.ones((bs, len(kpts0)), dtype=bool)

    # # Reject all correspondences outside the image plane
    # valid[keypoints_1[:, :, 0] < 0] = False
    # valid[keypoints_1[:, :, 1] < 0] = False
    # valid[keypoints_1[:, :, 0] > depth1.shape[2] - 1] = False
    # valid[keypoints_1[:, :, 1] > depth1.shape[1] - 1] = False

    # # Check if expected depth values match actual depth values

    # depth_values_1 = depth1[keypoints_1_rounded[valid][:, :, 1], keypoints_1_rounded[valid][:, :, 0]].reshape(bs, -1, 1)

    # delta_depth_values = (depth_values_1 - expected_depth_values_1[valid]).abs()

    # # Hacky way to set valid flag to false for all points for which there is a change in depth larger than a threshold
    # indices_of_not_valid = torch.arange(len(kpts0))
    # indices_of_not_valid = indices_of_not_valid[valid]
    # indices_of_not_valid = indices_of_not_valid[(delta_depth_values > 0.001).flatten()]
    # valid[indices_of_not_valid] = False

    # keypoints_1_rounded[torch.logical_not(valid)] = torch.tensor([-1, -1])

    # indices_of_valid = torch.arange(len(kpts0))
    # valid_keypoints_1 = keypoints_1_rounded[valid]
    # unique, counts = torch.unique(valid_keypoints_1, axis=0, return_counts=True)

    # if (counts > 1).any():
    #     for kpt in unique[counts > 1]:
    #         kpt_indices = indices_of_valid[(keypoints_1_rounded == kpt).all(axis=1)]
    #         idx_of_closest_kpt = kpt_indices[torch.argmin(residuals[kpt_indices])]
    #         keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = torch.tensor([-1, -1])
    #         valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False

    # return valid, keypoints_1_rounded

    ########################################################







    # Get the depth value of the keypoint

    # keypoints = kpts0.detach().clone()[0]
    # depth_image_0 = depth0.detach().clone()[0]
    # depth_image_1 = depth1.detach().clone()[0]
    # k0 = K0.detach().clone()[0]
    # k1 = K1.detach().clone()[0]
    # t0to1 = T_0to1.detach().clone()[0]

    # keypoints_long = keypoints.round().long()

    # depth_values_0 = depth_image_0[keypoints_long[:, 1], keypoints_long[:, 0]].reshape(-1, 1)

    # # print(f"Depth: {depth_values_0.shape}")
    # # print(f"0: {torch.ones_like(keypoints[:,0]).shape}")
    # # print(f"1: {keypoints.shape}")
    # # print(f"Kpts: {torch.cat((keypoints, torch.ones_like(keypoints[:,0])),dim=1).shape}")
    # # print(f"K: {k0.inverse().transpose(1,0).shape}")
    # # Get the position of the keypoint in the camera frame
    # keypoint_pos_C_0 = depth_values_0 * torch.cat((keypoints, torch.ones_like(keypoints[:,[0]])),
    #                                                 dim=1) @ k0.inverse().transpose(1,0)

    # # Calculate the coordinates of the new point in the second image.
    # keypoint_pos_C_1 = torch.cat((keypoint_pos_C_0, torch.ones_like(keypoint_pos_C_0[:,[0]])), dim=1) @ t0to1.transpose(1,0)

    # # Use this depth value to determine if a match was found
    # expected_depth_values_1 = keypoint_pos_C_1[:, 2:3].detach().clone()

    # # Project the point to image plane
    # keypoint_pos_C_1 = keypoint_pos_C_1[:, :3]
    # keypoint_pos_C_1 /= keypoint_pos_C_1[:, 2:3]

    # keypoints_1 = (keypoint_pos_C_1 @ k1.transpose(1,0))[:, :2]
    # keypoints_1_rounded = torch.round(keypoints_1).to(dtype=torch.int64)
    # residuals = keypoints_1 - keypoints_1_rounded
    # residuals = torch.sqrt((residuals * residuals).sum(1))

    # # Check which correspondences are valid
    # valid = torch.ones_like(keypoints[:,0], dtype=bool)
    # # print(valid.shape)
    # # print(keypoints_1.shape)

    # # Reject all correspondences outside the image plane
    # valid[keypoints_1[:, 0] < 0] = False
    # valid[keypoints_1[:, 1] < 0] = False
    # valid[keypoints_1[:, 0] > depth_image_1.shape[1] - 1] = False
    # valid[keypoints_1[:, 1] > depth_image_1.shape[0] - 1] = False

    # # Check if expected depth values match actual depth values

    # depth_values_1 = depth_image_1[keypoints_1_rounded[valid][:, 1], keypoints_1_rounded[valid][:, 0]].reshape(-1, 1)

    # delta_depth_values = (depth_values_1 - expected_depth_values_1[valid]).abs()

    # # Hacky way to set valid flag to false for all points for which there is a change in depth larger than a threshold
    # indices_of_not_valid = torch.arange(len(keypoints), device=kpts0.device)
    # indices_of_not_valid = indices_of_not_valid[valid]
    # indices_of_not_valid = indices_of_not_valid[(delta_depth_values > 0.001).flatten()]
    # valid[indices_of_not_valid] = False

    # keypoints_1_rounded[torch.logical_not(valid)] = torch.tensor([-1, -1], device=kpts0.device)

    # indices_of_valid = torch.arange(len(keypoints), device=kpts0.device)
    # valid_keypoints_1 = keypoints_1_rounded[valid]
    # unique, counts = torch.unique(valid_keypoints_1, dim=0, return_counts=True)
    # print(unique[counts > 1])

    # if (counts > 1).any():
    #     for kpt in unique[counts > 1]:
    #         kpt_indices = indices_of_valid[(keypoints_1_rounded == kpt).all(1)].cpu().numpy()
    #         idx_of_closest_kpt = kpt_indices[torch.argmin(residuals[kpt_indices])]
    #         # print(f"kpt_indices: {kpt_indices.shape}")
    #         # print(f"residuals: {residuals.shape}")
    #         # print(f"idx_of_closest_kpt: {idx_of_closest_kpt.shape}")
    #         # print(f"idx_of_closest_kpt actual: {idx_of_closest_kpt}")
    #         # print(f"keypoints_1_rounded: {keypoints_1_rounded.shape}")
    #         # print(f"List: {[idx for idx in kpt_indices if idx != idx_of_closest_kpt]}")
    #         keypoints_1_rounded[[idx for idx in kpt_indices if idx != idx_of_closest_kpt], :] = torch.tensor([-1, -1], device=kpts0.device)
    #         valid[[idx for idx in kpt_indices if idx != idx_of_closest_kpt]] = False
    
    # # print(torch.unsqueeze(valid, 0).shape)
    # # print(torch.unsqueeze(keypoints_1_rounded, 0).shape)
    # return torch.unsqueeze(valid, 0), torch.unsqueeze(keypoints_1, 0)
