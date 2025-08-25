#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# from hoi_scene.cameras import Camera
import numpy as np
# from general_utils import PILtoTorch
# from graphics_utils import fov2focal
from scipy.spatial.transform import Rotation
import open3d as o3d
import math
WARNED = False

# def loadCam(args, id, cam_info, resolution_scale):
#     orig_w, orig_h = cam_info['image'].size

#     if args.resolution in [1, 2, 4, 8]:
#         resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 3200:
#                 global WARNED
#                 if not WARNED:
#                     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                         "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = float(global_down) * float(resolution_scale)
#         resolution = (int(orig_w / scale), int(orig_h / scale))

#     resized_image_rgb = PILtoTorch(cam_info['image'], resolution)
#     resized_image_rgb_o = PILtoTorch(cam_info['image_o'], resolution)
#     resized_image_rgb_h= PILtoTorch(cam_info['image_h'], resolution)

#     gt_image = resized_image_rgb[:3, ...]
#     gt_image_o = resized_image_rgb_o[:3, ...]
#     gt_image_h = resized_image_rgb_h[:3, ...]
#     loaded_mask = None

#     if resized_image_rgb.shape[1] == 4:
#         loaded_mask = resized_image_rgb[3:4, ...]

#     if cam_info['bound_mask'] is not None:
#         resized_bound_mask = PILtoTorch(cam_info['bound_mask'], resolution)
#     else:
#         resized_bound_mask = None

#     if cam_info['bkgd_mask'] is not None:
#         resized_bkgd_mask = PILtoTorch(cam_info['bkgd_mask'], resolution)
#         resized_bkfd_mask_o=PILtoTorch(cam_info['bkgd_mask_o'], resolution)
#         resized_bkfd_mask_h=PILtoTorch(cam_info['bkgd_mask_h'], resolution)
#     else:
#         resized_bkgd_mask = None
#         resized_bkfd_mask_o=None
#         resized_bkfd_mask_h=None

#     return Camera(colmap_id=cam_info['uid'], R=cam_info['R'], T=cam_info['T'], K=cam_info['K'],
#                   FoVx=cam_info['FovX'], FoVy=cam_info['FovY'],
#                   image=gt_image, image_o=gt_image_o,image_h=gt_image_h,gt_alpha_mask=loaded_mask,
#                   image_name=cam_info['image_name'], uid=id,save_obj=cam_info['save_obj'],obj_faces=cam_info['obj_faces'],
#                   cam_trans=cam_info['cam_trans'],cam_param=cam_info['cam_param'],img_vis=cam_info['img_vis'],face_sim=cam_info['sim_faces'],
#                   bkgd_mask=resized_bkgd_mask,bkgd_mask_o=resized_bkfd_mask_o,bkgd_mask_h=resized_bkfd_mask_h,
#                   bound_mask=resized_bound_mask, smpl_param=cam_info['smpl_param'],
#                   world_vertex=cam_info['world_vertex'], world_bound=cam_info['world_bound'],
#                   big_pose_smpl_param=cam_info['big_pose_smpl_param'],
#                   big_pose_world_vertex=cam_info['big_pose_world_vertex'],
#                   big_pose_world_bound=cam_info['big_pose_world_bound'],
#                   data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

# def camera_to_JSON(id, camera : Camera):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = camera.R.transpose()
#     Rt[:3, 3] = camera.T
#     Rt[3, 3] = 1.0

#     W2C = np.linalg.inv(Rt)
#     pos = W2C[:3, 3]
#     rot = W2C[:3, :3]
#     serializable_array_2d = [x.tolist() for x in rot]
#     camera_entry = {
#         'id' : id,
#         'img_name' : camera.image_name,
#         'width' : camera.width,
#         'height' : camera.height,
#         'position': pos.tolist(),
#         'rotation': serializable_array_2d,
#         'fy' : fov2focal(camera.FovY, camera.height),
#         'fx' : fov2focal(camera.FovX, camera.width)
#     }
#     return camera_entry
def apply_transform_to_model(vertices, transform_matrix):
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])
    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]
def rotate_camera(rotation_angle, rotation_axis):
    camera_rotation_rad = math.radians(rotation_angle)
    if rotation_axis.lower() == 'y':
        camera_self_rotation = np.array([
            [math.cos(camera_rotation_rad), 0, math.sin(camera_rotation_rad)],
            [0, 1, 0],
            [-math.sin(camera_rotation_rad), 0, math.cos(camera_rotation_rad)]
        ])
    elif rotation_axis.lower() == 'x':
        camera_self_rotation = np.array([
            [1, 0, 0],
            [0, math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad)],
            [0, math.sin(camera_rotation_rad), math.cos(camera_rotation_rad)]
        ])
    elif rotation_axis.lower() == 'z':
        camera_self_rotation = np.array([
            [math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad), 0],
            [math.sin(camera_rotation_rad), math.cos(camera_rotation_rad), 0],
            [0, 0, 1]
        ])
    return camera_self_rotation
def transform_to_global(incam_params, global_params, hverts=None, overts=None):
    incam_orient, incam_trans = incam_params
    global_orient, global_trans = global_params
    axis_angle = incam_orient.detach().cpu().numpy()
    R_2ori = Rotation.from_rotvec(axis_angle).as_matrix().reshape(3, 3)
    T_2ori = incam_trans.detach().cpu().numpy().squeeze().copy()
    T_2ori[1] -= 0.7
    T_2ori[0] += 0.13

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_2ori.T
    transformation_matrix[:3, 3] = - R_2ori.T @ T_2ori
    if hverts is not None:
        hverts = apply_transform_to_model(hverts, transformation_matrix)
    if overts is not None:
        overts = apply_transform_to_model(overts, transformation_matrix)


    axis_angle = global_orient.cpu().numpy()
    global_R = Rotation.from_rotvec(axis_angle).as_matrix()
    global_T = global_trans.cpu().numpy().copy()
    global_T[0] -= 0.12
    global_T[1] -= 0.01
    global_T[2] += 0.03

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = global_R
    transformation_matrix[:3, 3] = global_T
    if hverts is not None:
        hverts = apply_transform_to_model(hverts, transformation_matrix)
    if overts is not None:
        overts = apply_transform_to_model(overts, transformation_matrix)
    return hverts, overts

def compute_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    return min_coords, max_coords, center, size

def compute_camera_position(bbox_center, bbox_size, ground_plane='xz', distance_factor=3.0):
    """
    计算相机位置，增加distance_factor参数控制距离
    distance_factor: 距离因子，越大相机越远
    """
    if ground_plane == 'xz':
        camera_y = bbox_center[1]
        max_dimension = max(bbox_size[0], bbox_size[2])
        distance = max_dimension * distance_factor  # 可调整的距离因子
        camera_x = bbox_center[0] + distance
        camera_z = bbox_center[2]
        camera_position = np.array([camera_x, camera_y, camera_z])
    return camera_position

def compute_camera_extrinsics(camera_position, target_position):
    """修复版本：正确计算外参矩阵"""
    # 计算相机朝向向量
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # 世界坐标系的上方向（假设Y轴向上）
    world_up = np.array([0, 1, 0])
    
    # 计算相机的右方向
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    # 重新计算上方向（确保正交）
    up = np.cross(right, forward)
    
    # 构建旋转矩阵 - 标准相机坐标系
    rotation_matrix = np.array([
        right,   # X轴：右方向
        -up,     # Y轴：下方向（图像坐标系Y向下）
        forward  # Z轴：前方向（朝向目标）
    ])
    
    # 构建外参矩阵 [R|t]
    extrinsics = np.zeros((3, 4))
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ camera_position
    
    return extrinsics, rotation_matrix, camera_position

def compute_camera_intrinsics(bbox_size, image_width, image_height, fov_degrees=60.0):
    """
    计算相机内参，允许自定义垂直视场角（FOV）。
    fov_degrees: 垂直视场角，以度为单位。
    """
    max_object_size = max(bbox_size)
    fov_y = np.radians(fov_degrees)  # 使用可配置的视场角
    focal_length_y = image_height / (2 * np.tan(fov_y / 2))
    focal_length_x = focal_length_y
    cx = image_width / 2
    cy = image_height / 2
    
    intrinsics = np.array([
        [focal_length_x, 0, cx],
        [0, focal_length_y, cy],
        [0, 0, 1]
    ])
    return intrinsics

def create_camera_for_object(vertices, image_width=800, image_height=600, ground_plane='xz', distance_factor=3.0, fov_degrees=60.0):

    min_coords, max_coords, center, size = compute_bounding_box(vertices)
    camera_position = compute_camera_position(center, size, ground_plane, distance_factor)
    extrinsics, rotation_matrix, camera_pos = compute_camera_extrinsics(camera_position, center)
    intrinsics = compute_camera_intrinsics(size, image_width, image_height, fov_degrees=fov_degrees)

    return {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'camera_position': camera_pos,
        'rotation_matrix': rotation_matrix,
        'target_position': center,
        'bounding_box': {
            'min': min_coords,
            'max': max_coords,
            'center': center,
            'size': size
        }
    }

