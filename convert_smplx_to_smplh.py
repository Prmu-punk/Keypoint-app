#!/usr/bin/env python3
"""
从transformed_parameters.json中选择一帧，将SMPL-X参数转换为SMPL-H格式，
并复制对应的帧图片和mask文件
"""

import json
import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
import pickle
import smplx

class SMPLXToSMPLHConverter:
    def __init__(self, smplh_model_path=None):
        # SMPL-H参数列表（相比SMPL-X去掉了expression, jaw_pose, leye_pose, reye_pose等面部参数）
        self.smplh_params = [
            'body_pose',
            'betas', 
            'global_orient',
            'transl',
            'left_hand_pose',
            'right_hand_pose'
        ]
        
        # 初始化SMPL-H模型
        if smplh_model_path is None:
            smplh_model_path = "video_optimizer/smpl_models/SMPLH_NEUTRAL.pkl"
        
        if os.path.exists(smplh_model_path):
            self.smplh_model = smplx.create(
                smplh_model_path,
                model_type='smplh',
                gender='neutral',
                num_betas=10,
                use_pca=False,
                flat_hand_mean=True
            )
            if torch.cuda.is_available():
                self.smplh_model = self.smplh_model.cuda()
            print(f"✓ Loaded SMPL-H model from: {smplh_model_path}")
        else:
            print(f"⚠ Warning: SMPL-H model not found at: {smplh_model_path}")
            self.smplh_model = None
    
    def load_transformed_parameters(self, file_path):
        """加载transformed_parameters.json文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def convert_smplx_to_smplh(self, smplx_params, frame_idx):
        """
        将指定帧的SMPL-X参数转换为SMPL-H格式
        
        Args:
            smplx_params: 完整的SMPL-X参数字典
            frame_idx: 要转换的帧索引
            
        Returns:
            smplh_params: 转换后的SMPL-H参数字典
        """
        frame_str = str(frame_idx)
        smplh_params = {}
        
        # 检查帧是否存在
        human_params = smplx_params.get('human_params', {})
        if not human_params:
            raise ValueError("No human_params found in transformed_parameters")
        
        # 检查第一个参数来确认帧是否存在
        first_param = list(human_params.keys())[0]
        if frame_str not in human_params[first_param]:
            available_frames = list(human_params[first_param].keys())
            raise ValueError(f"Frame {frame_idx} not found. Available frames: {available_frames[:10]}...")
        
        # 转换参数
        for param_name in self.smplh_params:
            if param_name in human_params and frame_str in human_params[param_name]:
                smplh_params[param_name] = human_params[param_name][frame_str]
                print(f"✓ Converted {param_name}: {len(smplh_params[param_name])} values")
            else:
                print(f"⚠ Warning: {param_name} not found for frame {frame_idx}")
        
        return smplh_params
    
    def generate_human_mesh(self, smplh_params):
        """
        使用SMPL-H参数生成人体mesh
        
        Args:
            smplh_params: SMPL-H参数字典
            
        Returns:
            trimesh.Trimesh: 生成的人体mesh
        """
        if self.smplh_model is None:
            raise ValueError("SMPL-H model not loaded")
        
        # 准备参数
        device = next(self.smplh_model.parameters()).device
        
        body_pose = torch.tensor(smplh_params['body_pose'], dtype=torch.float32, device=device).view(1, -1)
        betas = torch.tensor(smplh_params['betas'], dtype=torch.float32, device=device).view(1, -1)
        global_orient = torch.tensor(smplh_params['global_orient'], dtype=torch.float32, device=device).view(1, 3)
        transl = torch.tensor(smplh_params['transl'], dtype=torch.float32, device=device).view(1, 3)
        left_hand_pose = torch.tensor(smplh_params['left_hand_pose'], dtype=torch.float32, device=device).view(1, -1)
        right_hand_pose = torch.tensor(smplh_params['right_hand_pose'], dtype=torch.float32, device=device).view(1, -1)
        
        # 生成mesh
        with torch.no_grad():
            output = self.smplh_model(
                betas=betas,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                global_orient=global_orient,
                transl=transl
            )
        
        # 转换为trimesh
        vertices = output.vertices[0].cpu().numpy()
        faces = self.smplh_model.faces
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def load_object_mesh(self, object_mesh_path):
        """
        加载物体mesh
        
        Args:
            object_mesh_path: 物体mesh文件路径
            
        Returns:
            trimesh.Trimesh: 物体mesh
        """
        if not os.path.exists(object_mesh_path):
            raise FileNotFoundError(f"Object mesh not found: {object_mesh_path}")
        
        mesh = trimesh.load(object_mesh_path)
        return mesh
    
    def compute_curvature(self, mesh, curvature_type='mean'):
        """
        计算mesh的曲率
        
        Args:
            mesh: trimesh.Trimesh对象
            curvature_type: 曲率类型，'mean' 或 'gaussian'
            
        Returns:
            numpy.ndarray: 每个顶点的曲率值
        """
        try:
            if curvature_type == 'mean':
                curvatures = trimesh.curvature.discrete_mean_curvature_measure(
                    mesh, mesh.vertices, radius=0.01
                )
            elif curvature_type == 'gaussian':
                curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(
                    mesh, mesh.vertices, radius=0.01
                )
            else:
                raise ValueError(f"Unsupported curvature type: {curvature_type}")
            
            return curvatures
        except Exception as e:
            print(f"⚠ Warning: Failed to compute {curvature_type} curvature: {e}")
            # 返回零值作为fallback
            return np.zeros(len(mesh.vertices))
    
    def save_curvature_data(self, mesh, curvatures, output_path, mesh_type, frame_idx):
        """
        保存曲率数据
        
        Args:
            mesh: trimesh.Trimesh对象
            curvatures: 曲率数组
            output_path: 输出文件路径
            mesh_type: mesh类型 ('human' 或 'object')
            frame_idx: 帧索引
        """
        # 保存为numpy格式
        curvature_data = {
            'vertices': mesh.vertices,
            'faces': mesh.faces,
            'mean_curvature': curvatures,
            'metadata': {
                'mesh_type': mesh_type,
                'frame_index': frame_idx,
                'vertex_count': len(mesh.vertices),
                'face_count': len(mesh.faces),
                'curvature_stats': {
                    'min': float(np.min(curvatures)),
                    'max': float(np.max(curvatures)),
                    'mean': float(np.mean(curvatures)),
                    'std': float(np.std(curvatures))
                }
            }
        }
        
        # 保存为npz文件
        np.savez_compressed(output_path, **curvature_data)
        print(f"✓ Saved {mesh_type} curvature data to: {output_path}")
        
        # 同时保存为obj文件，用顶点颜色表示曲率
        obj_path = output_path.replace('.npz', '.obj')
        self.save_mesh_with_curvature_colors(mesh, curvatures, obj_path)
    
    def save_mesh_with_curvature_colors(self, mesh, curvatures, output_path):
        """
        保存带有曲率颜色的mesh
        
        Args:
            mesh: trimesh.Trimesh对象
            curvatures: 曲率数组
            output_path: 输出OBJ文件路径
        """
        # 归一化曲率值到[0, 1]
        curvatures_norm = curvatures - np.min(curvatures)
        if np.max(curvatures_norm) > 0:
            curvatures_norm = curvatures_norm / np.max(curvatures_norm)
        
        # 使用colormap将曲率值映射到颜色
        import matplotlib.cm as cm
        colormap = cm.get_cmap('viridis')  # 蓝色到黄色的颜色映射
        colors = colormap(curvatures_norm)[:, :3]  # 只取RGB，去掉alpha
        
        # 创建带颜色的mesh
        colored_mesh = mesh.copy()
        colored_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        
        # 保存
        colored_mesh.export(output_path)
        print(f"✓ Saved colored mesh to: {output_path}")
    
    def copy_frame_files(self, frame_idx, source_dirs, output_dir):
        """
        复制指定帧的图片和mask文件
        
        Args:
            frame_idx: 帧索引
            source_dirs: 源目录字典，包含frame_list, mask_dir, human_mask_dir的路径
            output_dir: 输出目录
        """
        frame_str = f"{frame_idx:05d}"  # 格式化为5位数字，如00001
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        copied_files = []
        
        # 复制原始帧图片
        # if 'frame_list' in source_dirs:
        frame_file = f"{frame_str}.png"
        src_path = os.path.join(source_dirs['frame_list'], frame_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, f"frame_{frame_str}.png")
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_path)
            print(f"✓ Copied frame: {frame_file} -> {dst_path}")
        else:
            print(f"⚠ Warning: Frame file not found: {src_path}")
        
        # 复制物体mask
        if 'mask_dir' in source_dirs:
            mask_file = f"{frame_str}_mask.png"
            src_path = os.path.join(source_dirs['mask_dir'], mask_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, f"object_mask_{frame_str}.png")
                shutil.copy2(src_path, dst_path)
                copied_files.append(dst_path)
                print(f"✓ Copied object mask: {mask_file} -> {dst_path}")
            else:
                print(f"⚠ Warning: Object mask file not found: {src_path}")
        
        # 复制人体mask
        if 'human_mask_dir' in source_dirs:
            human_mask_file = f"{frame_str}_mask.png"
            src_path = os.path.join(source_dirs['human_mask_dir'], human_mask_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, f"human_mask_{frame_str}.png")
                shutil.copy2(src_path, dst_path)
                copied_files.append(dst_path)
                print(f"✓ Copied human mask: {human_mask_file} -> {dst_path}")
            else:
                print(f"⚠ Warning: Human mask file not found: {src_path}")
        
        return copied_files
    
    def save_smplh_params(self, smplh_params, output_path, frame_idx, metadata=None):
        """
        保存SMPL-H参数到JSON文件
        
        Args:
            smplh_params: SMPL-H参数字典
            output_path: 输出文件路径
            frame_idx: 帧索引
            metadata: 额外的元数据
        """
        output_data = {
            'metadata': {
                'description': f'SMPL-H parameters converted from SMPL-X for frame {frame_idx}',
                'frame_index': frame_idx,
                'conversion_note': 'Removed SMPL-X specific parameters: expression, jaw_pose, leye_pose, reye_pose',
                'smplh_parameters': list(self.smplh_params)
            },
            'smplh_params': smplh_params
        }
        
        # 添加原始元数据
        if metadata:
            output_data['metadata']['original_metadata'] = metadata
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved SMPL-H parameters to: {output_path}")
    
    def process_frame(self, transformed_params_file, frame_idx, data_dir, output_dir, object_mesh_path=None):
        """
        处理指定帧：转换参数、复制文件、计算curvature
        
        Args:
            transformed_params_file: transformed_parameters.json文件路径
            frame_idx: 要处理的帧索引
            data_dir: 数据目录（包含frame_list, mask_dir, human_mask_dir）
            output_dir: 输出目录
            object_mesh_path: 物体mesh文件路径（可选）
        """
        # 加载参数文件
        print(f"Loading transformed parameters from: {transformed_params_file}")
        transformed_params = self.load_transformed_parameters(transformed_params_file)
        
        # 检查帧是否可用
        available_frames = transformed_params.get('metadata', {}).get('frame_indices', [])
        if frame_idx not in available_frames:
            print(f"⚠ Frame {frame_idx} not in available frames: {available_frames[:10]}...")
            return False
        
        print(f"Processing frame {frame_idx}...")
        
        # 转换SMPL-X参数为SMPL-H
        try:
            smplh_params = self.convert_smplx_to_smplh(transformed_params, frame_idx)
        except Exception as e:
            print(f"❌ Error converting parameters: {e}")
            return False
        
        # 设置源目录
        source_dirs = {
            'frame_list': os.path.join(data_dir, 'frame_list'),
            'mask_dir': os.path.join(data_dir, 'mask_dir'),
            'human_mask_dir': os.path.join(data_dir, 'human_mask_dir')
        }
        
        # 复制文件
        try:
            copied_files = self.copy_frame_files(frame_idx, source_dirs, output_dir)
        except Exception as e:
            print(f"❌ Error copying files: {e}")
            return False
        
        # 保存SMPL-H参数
        try:
            smplh_output_path = os.path.join(output_dir, f"smplh_params_frame_{frame_idx:05d}.json")
            self.save_smplh_params(
                smplh_params, 
                smplh_output_path, 
                frame_idx, 
                transformed_params.get('metadata')
            )
        except Exception as e:
            print(f"❌ Error saving SMPL-H parameters: {e}")
            return False
        
        # 计算人体mesh的curvature
        if self.smplh_model is not None:
            try:
                print("Generating human mesh and computing curvature...")
                human_mesh = self.generate_human_mesh(smplh_params)
                human_curvature = self.compute_curvature(human_mesh, 'mean')
                
                # 保存人体curvature数据
                human_curvature_path = os.path.join(output_dir, f"human_curvature_frame_{frame_idx:05d}.npz")
                self.save_curvature_data(human_mesh, human_curvature, human_curvature_path, 'human', frame_idx)
                
            except Exception as e:
                print(f"❌ Error computing human curvature: {e}")
        
        # 计算物体mesh的curvature
        if object_mesh_path and os.path.exists(object_mesh_path):
            try:
                print("Loading object mesh and computing curvature...")
                object_mesh = self.load_object_mesh(object_mesh_path)
                object_curvature = self.compute_curvature(object_mesh, 'mean')
                
                # 保存物体curvature数据
                object_curvature_path = os.path.join(output_dir, f"object_curvature_frame_{frame_idx:05d}.npz")
                self.save_curvature_data(object_mesh, object_curvature, object_curvature_path, 'object', frame_idx)
                
            except Exception as e:
                print(f"❌ Error computing object curvature: {e}")
        elif object_mesh_path:
            print(f"⚠ Warning: Object mesh not found: {object_mesh_path}")
        
        print(f"\n✅ Successfully processed frame {frame_idx}")
        print(f"   Output directory: {output_dir}")
        print(f"   Files created: {len(copied_files) + 1} (+ curvature files)")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Convert SMPL-X parameters to SMPL-H and copy frame files")
    # parser.add_argument('--transformed_params', type=str, default='data\chair\\final_optimized_parameters\\transformed_parameters_20250814_163943.json',
    parser.add_argument('--transformed_params', type=str, default='data\\umbrella\\final_optimized_parameters\\transformed_parameters_20250819_100923.json',
                       help='Path to transformed_parameters.json file')
    parser.add_argument('--frame_idx', type=int, default=20,
                       help='Frame index to process (default: 0)')
    # parser.add_argument('--data_dir', type=str, default='data/chair',
    parser.add_argument('--data_dir', type=str, default='data/umbrella',
                       help='Data directory containing frame_list, mask_dir, human_mask_dir')
    parser.add_argument('--output_dir', type=str, default='output/smplh_conversion',
                       help='Output directory for converted files')
    # parser.add_argument('--object_mesh', type=str, default='data/chair/obj_org.obj',
    parser.add_argument('--object_mesh', type=str, default='data/umbrella/obj_org.obj',
                       help='Path to object mesh file for curvature computation')
    parser.add_argument('--smplh_model', type=str, default='video_optimizer/smpl_models/SMPLH_NEUTRAL.pkl',
                       help='Path to SMPL-H model file')
    parser.add_argument('--list_frames', action='store_true',
                       help='List available frames and exit')
    
    args = parser.parse_args()
    
    # 如果没有指定transformed_params，自动寻找最新的
    if not args.transformed_params:
        params_dir = os.path.join(args.data_dir, 'final_optimized_parameters')
        if os.path.exists(params_dir):
            param_files = [f for f in os.listdir(params_dir) if f.startswith('transformed_parameters_') and f.endswith('.json')]
            if param_files:
                # 选择最新的文件
                param_files.sort(reverse=True)
                args.transformed_params = os.path.join(params_dir, param_files[0])
                print(f"Auto-selected parameters file: {args.transformed_params}")
            else:
                print("❌ No transformed_parameters files found in final_optimized_parameters directory")
                return
        else:
            print(f"❌ Parameters directory not found: {params_dir}")
            return
    
    if not os.path.exists(args.transformed_params):
        print(f"❌ Transformed parameters file not found: {args.transformed_params}")
        return
    
    converter = SMPLXToSMPLHConverter(args.smplh_model)
    
    # 如果只是列出可用帧
    if args.list_frames:
        try:
            params = converter.load_transformed_parameters(args.transformed_params)
            available_frames = params.get('metadata', {}).get('frame_indices', [])
            print(f"Available frames in {args.transformed_params}:")
            print(f"Total frames: {len(available_frames)}")
            print(f"Frame indices: {available_frames}")
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
        return
    
    # 处理指定帧
    success = converter.process_frame(
        args.transformed_params,
        args.frame_idx,
        args.data_dir,
        args.output_dir,
        args.object_mesh
    )
    
    if not success:
        print(f"❌ Failed to process frame {args.frame_idx}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
