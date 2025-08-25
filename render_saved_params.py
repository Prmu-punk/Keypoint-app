#!/usr/bin/env python3
"""
使用Renderer渲染保存的参数进行可视化
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import smplx
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
from einops import einsum, rearrange, repeat

# 导入渲染相关模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_optimizer.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from PIL import Image
import cv2

def get_writer(video_path, fps=30, crf=18):
    """创建视频写入器"""
    import imageio
    return imageio.get_writer(video_path, fps=fps, codec='libx264', quality=9, macro_block_size=1)

class ParameterVisualizer:
    def __init__(self, smpl_model_path=None):
        if smpl_model_path is None:
            smpl_model_path = "video_optimizer/smpl_models/SMPLX_NEUTRAL.npz"
        
        self.smpl_model = smplx.create(
            smpl_model_path, 
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False, 
            flat_hand_mean=True
        ).cuda()
        
        # if torch.cuda.is_available():
        #     self.smpl_model = self.smpl_model.cuda()
    
    def load_transformed_parameters(self, transformed_params_file):
        with open(transformed_params_file, 'r') as f:
            return json.load(f)
    
    def generate_human_meshes(self, human_params, frame_indices):
        all_vertices = []
        all_joints= []
        faces = None
        
        for frame_idx in frame_indices:
            frame_str = str(frame_idx)
            
            body_pose = torch.tensor(human_params['body_pose'][frame_str], dtype=torch.float32)
            betas = torch.tensor(human_params['betas'][frame_str], dtype=torch.float32) 
            global_orient = torch.tensor(human_params['global_orient'][frame_str], dtype=torch.float32)
            transl = torch.tensor(human_params['transl'][frame_str], dtype=torch.float32)
            left_hand_pose = torch.tensor(human_params['left_hand_pose'][frame_str], dtype=torch.float32)
            right_hand_pose = torch.tensor(human_params['right_hand_pose'][frame_str], dtype=torch.float32)
            
            zero_pose = torch.zeros((1, 3)).float()
            
            if torch.cuda.is_available():
                body_pose = body_pose.cuda()
                betas = betas.cuda()
                global_orient = global_orient.cuda()
                transl = transl.cuda()
                left_hand_pose = left_hand_pose.cuda()
                right_hand_pose = right_hand_pose.cuda()
                zero_pose = zero_pose.cuda()
            
            with torch.no_grad():
                output = self.smpl_model(
                    betas=betas.view(1, -1),
                    body_pose=body_pose.view(1, -1),
                    left_hand_pose=left_hand_pose.view(1, -1),
                    right_hand_pose=right_hand_pose.view(1, -1),
                    jaw_pose=zero_pose,
                    leye_pose=zero_pose,
                    reye_pose=zero_pose,
                    global_orient=global_orient.view(1, 3),
                    expression=torch.zeros((1, 10)).float().cuda() if torch.cuda.is_available() else torch.zeros((1, 10)).float(),
                    transl=transl.view(1, 3)
                )
            
            vertices = output.vertices[0]
            all_vertices.append(vertices)
            joint=output.joints[0]
            all_joints.append(joint)
            
            if faces is None:
                faces = self.smpl_model.faces
                # if torch.cuda.is_available():
                #     faces = faces.cuda()
        
        return torch.stack(all_vertices, dim=0), faces, torch.stack(all_joints, dim=0)
    def transform_mat(self, R, t):
        """
        Args:
            R: Bx3x3 array of a batch of rotation matrices
            t: Bx3x(1) array of a batch of translation vectors
        Returns:
            T: Bx4x4 Transformation matrix
        """
        # No padding left or right, only add an extra row
        if len(R.shape) > len(t.shape):
            t = t[..., None]
        return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)
    def compute_T_ayfz2ay(self,joints, inverse=False):
        """
        Args:
            joints: (B, J, 3), in the start-frame, ay-coordinate
        Returns:
            if inverse == False:
                T_ayfz2ay: (B, 4, 4)
            else :
                T_ay2ayfz: (B, 4, 4)
        """
        t_ayfz2ay = joints[:, 0, :].detach().clone()
        t_ayfz2ay[:, 1] = 0  # do not modify y

        RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side
        RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side
        RL_xz = RL_xz_h + RL_xz_s
        I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
        if I_mask.sum() > 0:
            Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))

        x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
        x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
        y_dir = torch.zeros_like(x_dir)
        y_dir[..., 1] = 1  # (B, 3)
        z_dir = torch.cross(x_dir, y_dir, dim=-1)
        R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
        R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)

        if inverse:
            R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
            t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")
            return self.transform_mat(R_ay2ayfz, t_ay2ayfz)
        else:
            return self.transform_mat(R_ayfz2ay, t_ayfz2ay)
    def apply_T_on_points(self,points, T):
        """
        Args:
            points: (..., N, 3)
            T: (..., 4, 4)
        Returns: (..., N, 3)
        """
        points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
        return points_T
    def render_combined_view(self, transformed_params_data, transformed_object_path, output_video_path, 
                           width=1024, height=1024, fps=30, crf=18):
        """渲染人体+物体组合视频"""
        
        frame_indices = transformed_params_data['metadata']['frame_indices']
        human_params = transformed_params_data['human_params']
        
        # 生成人体mesh
        verts_human, faces_human, joints_glob = self.generate_human_meshes(human_params, frame_indices)
        num_frames = len(frame_indices)
        print(joints_glob.shape)
        
        # 加载物体mesh
        object_mesh = o3d.io.read_triangle_mesh(transformed_object_path)
        if len(object_mesh.vertices) == 0:
            return
        
        # 处理物体mesh
        object_vertices = np.asarray(object_mesh.vertices)
        object_faces = np.asarray(object_mesh.triangles)
        object_color = np.asarray(object_mesh.vertex_colors)
        
        # 根据参数变换物体顶点
        verts_object = []
        if 'object_params' in transformed_params_data:
            object_params = transformed_params_data['object_params']
            for frame_idx in frame_indices:
                frame_str = str(frame_idx)
                
                # 获取变换参数
                if 'R_total' in object_params and 'T_total' in object_params:
                    R = np.array(object_params['R_total'][frame_str])
                    T = np.array(object_params['T_total'][frame_str])
                    
                    # 应用变换: vertices' = vertices @ R^T + T
                    transformed_verts = object_vertices @ R.T + T
                    verts_object.append(torch.tensor(transformed_verts, dtype=torch.float32))
                else:
                    # 如果没有变换参数，使用原始顶点
                    verts_object.append(torch.tensor(object_vertices, dtype=torch.float32))
        else:
            # 如果没有物体参数，使用原始顶点
            for frame_idx in frame_indices:
                verts_object.append(torch.tensor(object_vertices, dtype=torch.float32))
        
        verts_object = torch.stack(verts_object, dim=0)
        faces_object =object_faces
        
        if torch.cuda.is_available():
            verts_object = verts_object.cuda()
            # faces_object = faces_object.cuda()
        
        # 创建相机内参 - 提高焦距获得更清晰的渲染
        K = torch.tensor([
            [1200.0, 0.0, width/2],
            [0.0, 1200.0, height/2], 
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        if torch.cuda.is_available():
            K = K.cuda()
        
        # 初始化渲染器
        renderer = Renderer(width, height, device="cuda", faces_human=faces_human, faces_obj=faces_object, K=K)
        
        # 获取全局相机参数
        combined_verts = torch.cat([verts_human, verts_object], dim=1)
        # move verts to the ground
        offset=joints_glob.clone()[0][0]
        print(offset)
        offset[1]=combined_verts[:,:, [1]].min()
        print(offset)
        combined_verts -= offset
        joints_glob -= offset
        verts_human -= offset
        verts_object -= offset


        print(combined_verts[:,:, [1]].min())

        T_ay2ayfz = self.compute_T_ayfz2ay(joints_glob[[0]], inverse=True)
        combined_verts = self.apply_T_on_points(combined_verts, T_ay2ayfz)
        joints_glob = self.apply_T_on_points(joints_glob, T_ay2ayfz)
        verts_human = self.apply_T_on_points(verts_human, T_ay2ayfz)
        verts_object = self.apply_T_on_points(verts_object, T_ay2ayfz)

        global_R, global_T, global_lights = get_global_cameras_static(
            combined_verts.cpu(),
            beta=2.5,
            cam_height_degree=20,
            target_center_height=1.0,
            vec_rot=300
        )
        
        # 设置地面
        # joints_glob = verts_human.mean(dim=1, keepdim=True)
        scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], combined_verts)
        renderer.set_ground(scale * 1.5, cx, cz)
        
        # 渲染视频
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # 创建临时目录保存图片
        temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 渲染每一帧并保存图片
        frame_paths = []
        for i in tqdm(range(num_frames)):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img = renderer.render_with_ground_hoi(verts_human[i], verts_object[i], cameras, global_lights, [0.8, 0.8, 0.8], object_color)
            
            # 确保图像数据类型正确并保存为高质量PNG
            img = np.clip(img, 0, 255)
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            Image.fromarray(img.astype(np.uint8), mode='RGB').save(frame_path, optimize=False, compress_level=0)
            frame_paths.append(frame_path)
        
        # 将图片转换为视频
        writer = get_writer(output_video_path, fps=fps, crf=crf)
        try:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                img = Image.open(frame_path)
                img_array = np.array(img)
                writer.append_data(img_array)
        finally:
            writer.close()
        
        # # 清理临时文件
        # for frame_path in frame_paths:
        #     os.remove(frame_path)
        # os.rmdir(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="渲染保存的参数")
    parser.add_argument('--transformed_params', type=str,default='../visualization/transformed_parameters.json', help='参数JSON文件路径')
    parser.add_argument('--transformed_object', type=str,default='../visualization/transformed_object.obj', help='物体mesh文件路径')
    parser.add_argument('--output_video', type=str,default='./output_render.mp4', help='输出视频路径')
    parser.add_argument('--smpl_model', type=str,default='./video_optimizer/smpl_models/SMPLX_NEUTRAL.npz', help='SMPL模型路径')
    parser.add_argument('--width', type=int, default=1024, help='视频宽度')
    parser.add_argument('--height', type=int, default=1024, help='视频高度')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    parser.add_argument('--crf', type=int, default=18, help='压缩质量')
    
    args = parser.parse_args()
    
    visualizer = ParameterVisualizer(args.smpl_model)
    transformed_params_data = visualizer.load_transformed_parameters(args.transformed_params)
    
    visualizer.render_combined_view(
        transformed_params_data, 
        args.transformed_object, 
        args.output_video,
        args.width, 
        args.height, 
        args.fps, 
        args.crf
    )

if __name__ == "__main__":
    main()

