#!/usr/bin/env python3
"""
参数变换工具函数
用于将原始优化参数转换为最终的变换后参数
"""

import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import open3d as o3d
import os

def compute_combined_transform(incam_params, global_params):
    """
    计算组合的变换矩阵，将transform_to_global的变换整合为一个R和T
    :param incam_params: 相机内参数 (incam_orient, incam_trans)
    :param global_params: 全局参数 (global_orient, global_trans)
    :return: 组合的旋转矩阵R和平移向量T
    """
    incam_orient, incam_trans = incam_params
    global_orient, global_trans = global_params
    
    # 第一步变换：incam to original
    axis_angle = incam_orient.detach().cpu().numpy()
    R_2ori = Rotation.from_rotvec(axis_angle).as_matrix()
    T_2ori = incam_trans.detach().cpu().numpy().squeeze()
    T_2ori[1] -= 0.7
    T_2ori[0] += 0.13
    
    # 第二步变换：original to global
    axis_angle = global_orient.cpu().numpy()
    global_R = Rotation.from_rotvec(axis_angle).as_matrix()
    global_T = global_trans.cpu().numpy()
    global_T[0] -= 0.12
    global_T[1] -= 0.01
    global_T[2] += 0.03
    
    # 组合变换：先应用 incam->ori 变换，再应用 ori->global 变换
    # 对于旋转：R_combined = global_R @ R_2ori.T
    R_combined = global_R @ R_2ori.T
    
    # 对于平移：T_combined = global_R @ (-R_2ori.T @ T_2ori) + global_T
    T_combined = global_R @ (-R_2ori.T @ T_2ori) + global_T
    
    return R_combined, T_combined

def apply_transform_to_smpl_params(global_orient, transl, incam_params, global_params):
    """
    将transform_to_global的变换直接应用到SMPL参数上
    :param global_orient: 原始global_orient
    :param transl: 原始transl
    :param incam_params: 相机内参数
    :param global_params: 全局参数
    :return: 变换后的global_orient和transl
    """
    R_combined, T_combined = compute_combined_transform(incam_params, global_params)
    
    # 将组合变换应用到SMPL参数
    # 对于global_orient：需要将旋转组合
    original_orient = global_orient.cpu().numpy()
    original_R = Rotation.from_rotvec(original_orient).as_matrix()
    new_R = R_combined @ original_R
    new_orient = Rotation.from_matrix(new_R).as_rotvec()
    
    # 对于transl：应用组合变换
    original_transl = transl.cpu().numpy()
    if original_transl.ndim > 1:
        original_transl = original_transl.squeeze()
    new_transl = R_combined @ original_transl + T_combined
    
    return new_orient, new_transl

def transform_and_save_parameters(human_params_dict, org_params, camera_params, output_dir, original_object_path, user_scale=1.0):
    """
    变换并保存参数
    :param human_params_dict: 人体参数字典 {"body_pose": {...}, "global_orient": {...}, ...}
    :param org_params: 原始物体参数 (self.object_poses)
    :param camera_params: 相机参数 (self.output)  
    :param output_dir: 输出目录
    :param original_object_path: 原始物体mesh文件路径
    :param user_scale: 用户输入的scale参数
    :return: 保存成功的文件路径列表
    """
    print("🔄 Transforming and saving parameters...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 从相机参数中提取incam和global参数
    incam_params = camera_params["smpl_params_incam"]
    global_params = camera_params["smpl_params_global"]
    
    # 将字符串键转换为整数并排序
    sorted_frames = sorted([int(k) for k in human_params_dict['body_pose'].keys()])
    
    # 变换后的人体参数
    transformed_human_params = {
        'body_pose': {},
        'betas': {},
        'global_orient': {},
        'transl': {},
        'left_hand_pose': {},
        'right_hand_pose': {},
    }
    
    print(f"📝 Transforming human parameters for {len(sorted_frames)} frames...")
    
    for id, frame_idx in enumerate(sorted_frames):
        frame_str = str(frame_idx)
        
        # 获取原始参数
        original_global_orient = torch.tensor(human_params_dict['global_orient'][frame_str], dtype=torch.float32)
        original_transl = torch.tensor(human_params_dict['transl'][frame_str], dtype=torch.float32)
        
        # 获取相机参数
        incam_param = (incam_params['global_orient'][id], incam_params['transl'][id])
        global_param = (global_params['global_orient'][id], global_params['transl'][id])
        
        # 应用变换到SMPL参数
        new_global_orient, new_transl = apply_transform_to_smpl_params(
            original_global_orient, original_transl, incam_param, global_param
        )
        
        # 保存变换后的参数
        transformed_human_params['body_pose'][frame_str] = human_params_dict['body_pose'][frame_str]
        transformed_human_params['betas'][frame_str] = human_params_dict['betas'][frame_str]
        transformed_human_params['global_orient'][frame_str] = new_global_orient.tolist()
        transformed_human_params['transl'][frame_str] = new_transl.tolist()
        transformed_human_params['left_hand_pose'][frame_str] = human_params_dict['left_hand_pose'][frame_str]
        transformed_human_params['right_hand_pose'][frame_str] = human_params_dict['right_hand_pose'][frame_str]
    
    # 处理物体参数（如果存在物体优化结果）
    transformed_object_params = None
    transformed_object_path = None
    
    if 'poses' in org_params and org_params['poses'] and original_object_path and os.path.exists(original_object_path):
        print("📝 Transforming object parameters and mesh...")
        
        # 获取scale参数
        scale = org_params.get('scale', 1.0)
        scale_init = user_scale  # 用户输入的scale
        
        transformed_object_params = {
            'R_total': {},  # 最终旋转矩阵
            'T_total': {},  # 最终平移向量
            'scale': scale,
            'scale_init': scale_init
        }
        
        for id, frame_idx in enumerate(sorted_frames):
            frame_str = str(frame_idx)
            
            if frame_idx < len(org_params['poses']) and org_params['poses'][frame_str] is not None:
                # 获取原始物体变换
                R_final = np.array(org_params['poses'][frame_str])
                t_final = np.array(org_params['centers'][frame_str])
                
                # 获取相机参数
                incam_param = (incam_params['global_orient'][id], incam_params['transl'][id])
                global_param = (global_params['global_orient'][id], global_params['transl'][id])
                
                # 计算组合变换矩阵
                R_combined, T_combined = compute_combined_transform(incam_param, global_param)
                
                # 计算最终变换矩阵
                R_total = R_combined @ R_final
                T_total = R_combined @ t_final + T_combined
                
                # 保存最终变换
                transformed_object_params['R_total'][frame_str] = R_total.tolist()
                transformed_object_params['T_total'][frame_str] = T_total.tolist()
            else:
                # 如果某帧没有物体参数，使用单位矩阵和零向量
                transformed_object_params['R_total'][frame_str] = np.eye(3).tolist()
                transformed_object_params['T_total'][frame_str] = np.zeros(3).tolist()
        
        # 变换并保存物体mesh
        transformed_object_path = transform_and_save_object_mesh(
            original_object_path, scale, scale_init, output_dir
        )
        
        print(f"✅ Transformed object parameters with scale: {scale}, scale_init: {scale_init}")
    
    # 生成时间戳
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存变换后的参数
    saved_files = []
    
    # 保存完整的变换后参数
    transformed_data = {
        'metadata': {
            'description': 'Transformed parameters with all transforms applied',
            'total_frames': len(sorted_frames),
            'frame_indices': sorted_frames,
            'has_object_params': transformed_object_params is not None,
            'user_scale': user_scale
        },
        'human_params': transformed_human_params
    }
    
    if transformed_object_params is not None:
        transformed_data['object_params'] = transformed_object_params
    
    transformed_params_path = os.path.join(output_dir, f'transformed_parameters_{timestamp}.json')
    with open(transformed_params_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    saved_files.append(transformed_params_path)
    print(f"✅ Saved transformed parameters: {transformed_params_path}")
    
    # 如果有变换后的物体mesh，添加到文件列表
    if transformed_object_path:
        saved_files.append(transformed_object_path)
    
    print(f"🎉 Successfully saved {len(saved_files)} files!")
    return saved_files

def transform_and_save_object_mesh(original_object_path, scale, scale_init, output_dir):
    """
    对物体mesh应用scale变换并保存新的物体mesh
    """
    print(f"🔄 Transforming object mesh: {original_object_path}")
    
    # 加载原始物体mesh
    original_mesh = o3d.io.read_triangle_mesh(original_object_path)
    if len(original_mesh.vertices) == 0:
        print(f"❌ Error: Could not load mesh from {original_object_path}")
        return None
    
    original_vertices = np.asarray(original_mesh.vertices)
    
    # 应用scale变换
    print(f"📏 Applying scale transformations: scale={scale}, scale_init={scale_init}")
    scaled_vertices = original_vertices * scale
    
    # 中心化并应用初始scale
    center_m = np.mean(scaled_vertices, axis=0)
    scaled_vertices -= center_m  # 中心化顶点
    scaled_vertices *= scale_init  # 恢复初始scale
    
    # 创建新的mesh
    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
    transformed_mesh.triangles = original_mesh.triangles
    
    # 如果原始mesh有颜色，保持颜色
    if original_mesh.has_vertex_colors():
        transformed_mesh.vertex_colors = original_mesh.vertex_colors
    if original_mesh.has_vertex_normals():
        transformed_mesh.vertex_normals = original_mesh.vertex_normals
    
    # 重新计算法向量
    transformed_mesh.compute_vertex_normals()
    
    # 生成时间戳和输出路径
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_object_path = os.path.join(output_dir, f'transformed_object_{timestamp}.obj')
    
    # 保存变换后的mesh
    success = o3d.io.write_triangle_mesh(output_object_path, transformed_mesh)
    
    if success:
        print(f"✅ Transformed object mesh saved to: {output_object_path}")
        return output_object_path
    else:
        print(f"❌ Error: Failed to save transformed mesh to: {output_object_path}")
        return None
