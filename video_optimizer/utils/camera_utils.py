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
from general_utils import PILtoTorch
from graphics_utils import fov2focal
from scipy.spatial.transform import Rotation
import open3d as o3d

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

def transform_to_global(hverts, overts, incam_params, global_params):
    incam_orient, incam_trans = incam_params
    global_orient, global_trans = global_params
    axis_angle = incam_orient.detach().cpu().numpy()
    R_2ori = Rotation.from_rotvec(axis_angle).as_matrix()
    T_2ori = incam_trans.detach().cpu().numpy().squeeze()
    T_2ori[1] -= 0.7
    T_2ori[0] += 0.13

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_2ori.T
    transformation_matrix[:3, 3] = - R_2ori.T @ T_2ori

    hverts = apply_transform_to_model(human_verts, transformation_matrix)
    overts = apply_transform_to_model(object_vertices, transformation_matrix)


    axis_angle = global_orient.cpu().numpy()
    global_R = Rotation.from_rotvec(axis_angle).as_matrix()
    global_T = global_trans.cpu().numpy()
    global_T[0] -= 0.12
    global_T[1] -= 0.01
    global_T[2] += 0.03

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = global_R
    transformation_matrix[:3, 3] = global_T
    human_verts = apply_transform_to_model(hverts, transformation_matrix)
    object_vertices = apply_transform_to_model(overts, transformation_matrix)
    return human_verts, object_vertices

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

def compute_camera_intrinsics(bbox_size, image_width, image_height, fov_factor=1.0):
    max_object_size = max(bbox_size)
    fov_y = np.radians(60)  # 60度视场角
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

def create_camera_for_object(vertices, image_width=800, image_height=600, ground_plane='xz', distance_factor=3.0):
    """
    添加distance_factor参数来控制相机距离
    distance_factor=2.0: 较近
    distance_factor=3.0: 中等距离（默认）
    distance_factor=4.0: 较远
    distance_factor=5.0: 很远
    """
    min_coords, max_coords, center, size = compute_bounding_box(vertices)
    camera_position = compute_camera_position(center, size, ground_plane, distance_factor)
    extrinsics, rotation_matrix, camera_pos = compute_camera_extrinsics(camera_position, center)
    intrinsics = compute_camera_intrinsics(size, image_width, image_height)
    
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

def test_camera():
    """测试相机函数"""
    
    # 创建测试数据
    hverts = np.array([
        [-0.2, 1.8, -0.2], [0.2, 1.8, -0.2], [0.2, 2.0, -0.2], [-0.2, 2.0, -0.2],
        [-0.2, 1.8, 0.2], [0.2, 1.8, 0.2], [0.2, 2.0, 0.2], [-0.2, 2.0, 0.2],
        [-0.3, 0.0, -0.15], [0.3, 0.0, -0.15], [0.3, 1.8, -0.15], [-0.3, 1.8, -0.15],
        [-0.3, 0.0, 0.15], [0.3, 0.0, 0.15], [0.3, 1.8, 0.15], [-0.3, 1.8, 0.15],
    ])
    
    overts = np.array([
        [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 1.0, 2.0], [1.0, 1.0, 2.0],
    ])
    
    vertices = np.concatenate([hverts, overts], axis=0)
    
    # 调整这里的distance_factor来控制相机距离
    # 可以尝试 4.0, 5.0, 6.0 等更大的值
    camera_params = create_camera_for_object(vertices, distance_factor=4.0)
    
    print("=== 相机参数调试 ===")
    print(f"相机位置: {camera_params['camera_position']}")
    print(f"目标位置: {camera_params['target_position']}")
    print(f"包围盒大小: {camera_params['bounding_box']['size']}")
    
    # 计算相机到目标的距离
    distance = np.linalg.norm(camera_params['camera_position'] - camera_params['target_position'])
    print(f"相机距离: {distance:.2f}")
    
    # 转换为Open3D相机
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(800, 600,
                           camera_params['intrinsics'][0,0],
                           camera_params['intrinsics'][1,1],
                           camera_params['intrinsics'][0,2],
                           camera_params['intrinsics'][1,2])
    
    extrinsic = np.eye(4)
    extrinsic[:3, :] = camera_params['extrinsics']
    
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = intrinsic
    camera.extrinsic = extrinsic
    
    # 创建几何体
    human_mesh = o3d.geometry.TriangleMesh.create_box(0.6, 2.0, 0.4)
    human_mesh.translate([-0.3, 0, -0.2])
    human_mesh.paint_uniform_color([0, 1, 0])
    human_mesh.compute_vertex_normals()
    
    object_mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    object_mesh.translate([1.0, 0, 1.0])  
    object_mesh.paint_uniform_color([1, 0, 0])
    object_mesh.compute_vertex_normals()
    
    ground = o3d.geometry.TriangleMesh.create_box(4.0, 0.1, 4.0)
    ground.translate([-1.0, -0.1, -1.0])
    ground.paint_uniform_color([0.5, 0.5, 0.5])
    ground.compute_vertex_normals()
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    # 创建相机位置标记
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    camera_marker.translate(camera_params['camera_position'])
    camera_marker.paint_uniform_color([1, 0, 1])  # 紫色
    camera_marker.compute_vertex_normals()
    
    # 创建目标标记
    target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    target_marker.translate(camera_params['target_position'])
    target_marker.paint_uniform_color([1, 1, 0])  # 黄色
    target_marker.compute_vertex_normals()
    
    # 启动可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, window_name="Camera Test - Distance Factor 4.0")
    
    vis.add_geometry(human_mesh)
    vis.add_geometry(object_mesh)
    vis.add_geometry(ground)
    vis.add_geometry(coord_frame)
    vis.add_geometry(camera_marker)
    vis.add_geometry(target_marker)
    
    # 先显示默认视角
    print("\n显示默认视角（3秒后切换到相机视角）")
    vis.poll_events()
    vis.update_renderer()
    
    import time
    for i in range(3, 0, -1):
        print(f"切换倒计时: {i}秒")
        time.sleep(1)
    
    # 应用相机参数
    print("应用相机参数...")
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
    
    vis.poll_events()
    vis.update_renderer()
    
    print("渲染窗口已打开，按Q键关闭")
    print(f"当前距离因子: 4.0，相机距离: {distance:.2f}")
    print("可以尝试的距离因子:")
    print("- 2.0: 很近")
    print("- 3.0: 中等") 
    print("- 4.0: 较远（当前）")
    print("- 5.0: 很远")
    print("- 6.0: 超远")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    test_camera()