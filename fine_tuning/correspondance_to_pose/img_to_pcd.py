import numpy as np
import open3d as o3d

def intrinsic_matrix_to_o3d(intrinsic_matrix: np.ndarray, image_h=None, image_w=None):
    intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix

    if image_w is not None:
        intrinsic_matrix_o3d.width = image_w
    if image_h is not None:
        intrinsic_matrix_o3d.height = image_h

    return intrinsic_matrix_o3d

def get_pixel_indices_from_segmap(segmap):
    """
    Return x, y
    """
    x = np.arange(segmap.shape[1])  # TODO should we have '+ 0.5' here?
    y = np.arange(segmap.shape[0])  # TODO should we have '+ 0.5' here?
    xx, yy = np.meshgrid(x, y)
    indices = np.concatenate((xx[..., None], yy[..., None]), axis=2)
    return indices[segmap]


def get_3d_points_from_pixel_indices(pixel_indices, intrinsic_matrix, depth_map, depth_units: str = 'mm'):
    assert pixel_indices.shape[1] == 2
    assert pixel_indices.dtype == np.int64
    assert depth_units in ['m', 'mm'], 'Depth units must be either meters or millimeters'

    depth_values = depth_map[pixel_indices[:, 1], pixel_indices[:, 0]].reshape(-1, 1)
    if depth_units == 'mm':
        depth_values = depth_values / 1e3
    homogenous_pixel_indices = np.concatenate((pixel_indices, np.ones((len(pixel_indices), 1))), axis=1)
    return depth_values * homogenous_pixel_indices @ np.linalg.inv(intrinsic_matrix).T


def img_to_o3d_pcd(depth: np.ndarray, intrinsic_matrix: np.ndarray, rgb=None):
    if depth.dtype == np.int32:
        depth = depth.astype(np.uint16)
    assert depth.dtype == np.uint16, f'The depth image must be \'mm\' stored in the dtype np.uint16 and not {depth.dtype}'
    if rgb is not None:
        assert rgb.dtype == np.uint8, f'The RGB image must be stored in the dtype np.uint8 and not {depth.rgb}'

    intrinsic_matrix_o3d = intrinsic_matrix_to_o3d(intrinsic_matrix)

    depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))  # On lab pc it is necessary to call .astype(np.uint16)

    if rgb is not None:
        rgb_o3d = o3d.geometry.Image(rgb)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_o3d, depth=depth_o3d,
                                                                      depth_scale=1000,
                                                                      convert_rgb_to_intensity=False)
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                 extrinsic=np.eye(4))
    else:
        pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic_matrix_o3d,
                                                                  extrinsic=np.eye(4))

    return pcd_o3d
