import numpy as np  
import matplotlib.pyplot as plt  
import math
import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch  
import torch.optim as optim  
import torch.nn.functional as F   
import smplx  
import json  
import cv2  
import open3d as o3d  
from torch import nn  
import tqdm
from .utils.loss_utils import HOCollisionLoss  
from pytorch3d.structures import Meshes  
from einops import einsum, rearrange
from PIL import Image
from pytorch3d.renderer.cameras import FoVPerspectiveCameras     
from tqdm import tqdm  
from .utils.rotate_smpl import matrix_to_axis_angle
from torchviz import make_dot
from .hoi_solver import HOISolver
from pytorch3d.renderer import (  
    RasterizationSettings,  
    MeshRasterizer,  
    BlendParams,  
    PointLights,  
    SoftSilhouetteShader  
)
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from .utils.image_utils import process_frame2square, process_frame2square_mask
# import neural_renderer as nr
import torchvision.transforms.functional as TF
import time
from scipy.ndimage.morphology import distance_transform_edt
from pykalman import KalmanFilter
from .utils.camera_utils import create_camera_for_object, transform_to_global
import gtsam
from .utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

def resource_path(relative_path):
    try:
        # PyInstaller创建的临时文件夹路径
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境下的路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

J_regressor = torch.load(resource_path("video_optimizer/J_regressor.pt")).float().cuda()
def load_downsampling_mapping(filepath):
    data = np.load(filepath)
    from scipy.sparse import csr_matrix
    D = csr_matrix((data['D_data'], data['D_indices'], data['D_indptr']), 
                   shape=data['D_shape'])
    faces_ds = data['faces_ds']
    print(f"Downsampling mapping loaded from {filepath}")
    return D, faces_ds

downsampling_file_path = resource_path("video_optimizer/smplx_downsampling_1000.npz")
D, faces_ds = load_downsampling_mapping(downsampling_file_path)
D_torch = torch.tensor(D.toarray(), dtype=torch.float32, device="cuda")
class VideoBodyObjectOptimizer:  
    def __init__(self,   
                 body_params,
                 global_body_params,
                 hand_params,  
                 object_points_idx,   
                 body_points_idx, 
                 pairs_2d, 
                 object_meshes, 
                 sampled_obj_meshes,
                #  centers,
                 initial_R,
                 initial_t,
                 transform_matrix,
                 smpl_model,
                 start_frame,
                 end_frame,
                 video_dir,  
                 lr=0.1,
                 is_static_object=False):  
        """  
        :param body_params: incam对应的人体参数 (['body_pose', 'betas', 'global_orient', 'transl'])  
        :param object_poses: 初始物体位姿路径，每帧一个  
        :param object_points_idx: 物体对应点的索引  
        :param body_points_idx: 人体对应点的索引  
        :param object_path: 物体网格路径  
        :param smpl_model: SMPL模型  
        :param seq_length: 序列长度  
        :param lr: 学习率  
        """  
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.seq_length = end_frame - start_frame
        self.smpl_model = smpl_model  
        self.pairs_2d = pairs_2d
        self.body_params_sequence = body_params  
        self.global_body_params = global_body_params
        self.hand_poses = hand_params
        self.object_meshes = object_meshes
        self.sampled_obj_meshes = sampled_obj_meshes
        # self.centers = centers
        self.initial_R = initial_R
        self.centers = initial_t
        self.transform_matrix = transform_matrix
        self.video_dir = video_dir
        self.is_static_object = is_static_object

        cap = cv2.VideoCapture(os.path.join(video_dir, "video.mp4"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = width
        self.height = height
        self.image_size = max(width, height)
        self.lr = lr
        # self.obj_scale = nn.Parameter(torch.tensor(1.0).float().cuda(), requires_grad=True)
        self.hoi_solver =  HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

        
        # learnable parameters  
        self.body_pose_params = []  
        self.shape_params = []
        self.left_hand_params = []
        self.right_hand_params = []
        self.global_orient = []
        self.transl = []
        self.obj_x_params = []  
        self.obj_y_params = []  
        self.obj_z_params = []  
        self.obj_transl_params = []  
        self.obj_transl_limit = torch.tensor([0.1, 0.1, 0.1]).cuda()

        self.base_obj_R = []
        self.base_obj_transl = []
        
        for i in range(self.start_frame, self.end_frame):  
            self.body_pose_params.append(nn.Parameter(self.body_params_sequence["body_pose"][i].cuda(), requires_grad=True)) 
            self.shape_params.append(nn.Parameter(self.body_params_sequence['betas'][i].cuda(), requires_grad=True))  
            handpose=self.hand_poses[str(i)]
            ## boran update
            left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1,3)).float().cuda()
            right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1,3)).float().cuda()
            self.left_hand_params.append(nn.Parameter(left_hand_pose, requires_grad=True))
            self.right_hand_params.append(nn.Parameter(right_hand_pose, requires_grad=True))
            self.global_orient.append(self.body_params_sequence['global_orient'][i].cuda())
            self.transl.append(self.body_params_sequence['transl'][i].cuda())

            trans_mat = self.transform_matrix[i - self.start_frame]
            R_mat = trans_mat[:3, :3]
            transl_vec = trans_mat[:3, 3]
            
            self.base_obj_R.append(torch.from_numpy(R_mat).float().cuda())
            self.base_obj_transl.append(torch.from_numpy(transl_vec).float().cuda())

            self.obj_x_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_y_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_z_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_transl_params.append(nn.Parameter(torch.zeros(3, dtype=torch.float32).cuda(), requires_grad=True))

        self.body_points_idx = body_points_idx
        self.object_points_idx = object_points_idx  

    
          
        self.mask = None  
        self.optimizer = None  
        
        # 当前帧
        self.current_frame = 0  
    
    def set_mask(self, joints_to_optimize):
        """
        设置掩码以仅优化指定的SMPL-X关节
        :param joints_to_optimize: 需要优化的关节名称列表
        """
        # 21*3, 15*3, 15*3
        body_mask = torch.zeros(63, dtype=torch.bool).cuda()
        left_hand_mask = torch.zeros((15, 3), dtype=torch.bool).cuda()
        right_hand_mask = torch.zeros((15, 3), dtype=torch.bool).cuda()

        body_joints = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
            "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist"
        ]
        left_hand_joints = [
            "left_thumb1", "left_thumb2", "left_thumb3",
            "left_index1", "left_index2", "left_index3",
            "left_middle1", "left_middle2", "left_middle3",
            "left_ring1", "left_ring2", "left_ring3",
            "left_pinky1", "left_pinky2", "left_pinky3"
        ]
        right_hand_joints = [
            "right_thumb1", "right_thumb2", "right_thumb3",
            "right_index1", "right_index2", "right_index3",
            "right_middle1", "right_middle2", "right_middle3",
            "right_ring1", "right_ring2", "right_ring3",
            "right_pinky1", "right_pinky2", "right_pinky3"
        ]

        for joint in joints_to_optimize:
            if joint in body_joints:
                idx = body_joints.index(joint)
                body_mask[idx*3:(idx+1)*3] = True
            elif joint in left_hand_joints:
                idx = left_hand_joints.index(joint)
                left_hand_mask[idx, :] = True
            elif joint in right_hand_joints:
                idx = right_hand_joints.index(joint)
                right_hand_mask[idx, :] = True

        self.body_mask = body_mask
        self.left_hand_mask = left_hand_mask
        self.right_hand_mask = right_hand_mask
    
    def training_setup(self):  
        """设置优化器参数"""  
        params_list = []  
        
        # 为序列中的每一帧添加参数（静态物体只有一帧）
        for i in range(self.seq_length):  
            frame_params = [
                {'params': [self.body_pose_params[i]], 'lr': 0.001, 'name': f'pose_{i}'},  
                {'params': [self.shape_params[i]], 'lr': 0.001, 'name': f'shape_{i}'},
                {'params': [self.left_hand_params[i]], 'lr': 0.003, 'name': f'left_hand_{i}'},
                {'params': [self.right_hand_params[i]], 'lr': 0.003, 'name': f'right_hand_{i}'},
                {'params': [self.obj_x_params[i]], 'lr': 0.002, 'name': f'x_angle_{i}'},  
                {'params': [self.obj_y_params[i]], 'lr': 0.002, 'name': f'y_angle_{i}'},  
                {'params': [self.obj_z_params[i]], 'lr': 0.002, 'name': f'z_angle_{i}'},  
                {'params': [self.obj_transl_params[i]], 'lr': 0.001, 'name': f'transl_{i}'},             
            ]
            
            params_list.extend(frame_params)
        
        # params_list.extend([{'params': [self.obj_scale], 'lr': 0.001, 'name': f'obj_scale'}])
        self.optimizer = optim.Adam(params_list, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
   

    def get_body_points(self, frame_idx=None, sampled=False):  
        """  
        获取指定帧的SMPL人体点  
        :param frame_idx: 帧索引，默认为当前帧  
        :return: 人体顶点坐标  
        """  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cuda()  
        shape = self.shape_params[frame_idx].reshape(1, -1).cuda()  
        global_orient = self.global_orient[frame_idx].reshape(1, 3).cuda()
        left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cuda()
        right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cuda()
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
        transl = self.transl[frame_idx].reshape(1, -1).cuda()
        # print(global_orient.shape ,body_pose.shape, left_hand_pose.shape ,right_hand_pose.shape ,zero_pose.shape)
        # if smpl
        output = self.smpl_model(betas=shape,   
                                body_pose=body_pose,
                                left_hand_pose=left_hand_pose,   
                                right_hand_pose=right_hand_pose,   
                                jaw_pose=zero_pose,   
                                leye_pose=zero_pose,
                                reye_pose=zero_pose,
                                global_orient=global_orient,
                                expression=torch.zeros((1, 10)).float().cuda(),
                                transl=transl)

        # smpl_poses = torch.cat([body_pose, left_hand_pose[:,:3], right_hand_pose[:,:3]], dim=1)
        # output = self.smpl_model(  
        #     betas=shape,  
        #     body_pose=smpl_poses, 
        #     global_orient=global_orient, 
        #     transl=transl  
        # )  
        xyz = output.vertices[0]
        if sampled:
            xyz = torch.einsum('vw,wc->vc', D_torch, xyz)
        return xyz
    
    def get_body_faces(self, sampled=False):  
        """获取人体网格面片"""  
        body_faces = self.smpl_model.faces  
        if sampled:
            body_faces = faces_ds
        return body_faces  
    
    def get_object_faces(self, frame_idx=None, sampled=False):
        """
        获取物体网格面片
        :param frame_idx: 帧索引，默认为当前帧
        :param sampled: 是否使用采样后的mesh
        :return: 物体面片数组 (int64格式)
        """
        if frame_idx is None:
            frame_idx = self.current_frame
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]
        object_faces = object_mesh.triangles
        return np.asarray(object_faces).astype(np.int64)
  
    
    
    def get_object_points(self, frame_idx=None, sampled=False):
        """
        获取指定帧的变换后物体顶点
        :param frame_idx: 帧索引，默认为当前帧
        :param sampled: 是否使用采样后的mesh
        :return: 变换后的物体顶点坐标
        """
        if frame_idx is None:
            frame_idx = self.current_frame
        if self.is_static_object:
            frame_idx = 0
        
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]

        R_final, t_final = self.get_object_params(frame_idx)
        
        object_points = torch.tensor(np.asarray(object_mesh.vertices), 
                                   dtype=torch.float32, device=R_final.device)

        # v' = v @ R_final.T + t_final
        object_points = torch.mm(object_points, R_final.T) + t_final
        
        return object_points
    
  
    
    def get_object_transform(self, frame_idx=None):  
        """  
        获取指定帧的物体变换矩阵  
        :param frame_idx: 帧索引，默认为当前帧  
        :return: 3x3旋转矩阵  
        """  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        
        x_angle = torch.deg2rad(self.obj_x_params[frame_idx])  
        y_angle = torch.deg2rad(self.obj_y_params[frame_idx])  
        z_angle = torch.deg2rad(self.obj_z_params[frame_idx]) 
        
        RX = torch.stack([  
            torch.tensor([1.0, 0.0, 0.0], device=x_angle.device, dtype=torch.float32),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.cos(x_angle), -torch.sin(x_angle)]),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.sin(x_angle), torch.cos(x_angle)])  
        ])  
        
        RY = torch.stack([  
            torch.stack([torch.cos(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.sin(y_angle)]),  
            torch.tensor([0.0, 1.0, 0.0], device=y_angle.device, dtype=torch.float32),  
            torch.stack([-torch.sin(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.cos(y_angle)])  
        ])  
        
        RZ = torch.stack([  
            torch.stack([torch.cos(z_angle), -torch.sin(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.stack([torch.sin(z_angle), torch.cos(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.tensor([0.0, 0.0, 1.0], device=z_angle.device, dtype=torch.float32)  
        ])  
        
        # 组合旋转矩阵  
        R = torch.mm(torch.mm(RZ, RY), RX)  
        return R  
    
    def get_optimized_parameters(self):
        """
        获取优化后的所有参数
        :return: 包含人体参数和物体参数的字典
        """
        human_params = {
            'body_pose': [],
            'betas': [],
            'global_orient': [],
            'transl': [],
            'left_hand_pose': [],
            'right_hand_pose': [],
        }
        
        object_params = {
            'poses': [],  # R_final
            'centers': []  # t_final
        }
        
        # 收集所有帧的参数
        for frame_idx in range(self.seq_length):
            # 人体参数
            body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            shape = self.shape_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            global_orient = self.global_orient[frame_idx].reshape(1, 3).cpu().detach().numpy()
            left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            transl = self.transl[frame_idx].reshape(1, -1).cpu().detach().numpy()
            
            human_params['body_pose'].append(body_pose.tolist())
            human_params['betas'].append(shape.tolist())
            human_params['global_orient'].append(global_orient.tolist())
            human_params['transl'].append(transl.tolist())
            human_params['left_hand_pose'].append(left_hand_pose.tolist())
            human_params['right_hand_pose'].append(right_hand_pose.tolist())
            
            # 物体参数 - 获取R_final和t_final
            if self.is_static_object:
                obj_frame_idx = 0
            else:
                obj_frame_idx = frame_idx
                
            # 计算R_final和t_final (参考get_object_points方法)
            R_final, t_final = self.get_object_params(obj_frame_idx)
            
            object_params['poses'].append(R_final.cpu().detach().numpy().tolist())
            object_params['centers'].append(t_final.cpu().detach().numpy().tolist())
        
        return {
            'human_params': human_params,
            'object_params': object_params,
            'frame_range': {
                'start_frame': self.start_frame,
                'end_frame': self.end_frame
            }
        }
    
    def get_corresponding_point(self, frame_idx=None):  
        """  
        获取指定帧的对应点  
        :param frame_idx: 帧索引，默认为当前帧  
        :return: 人体点和物体点的对应关系字典  
        """  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        
        # 获取在交互的人体点索引 

        object_points_idx =  self.object_points_idx[frame_idx]
        body_points_idx = np.asarray(self.body_points_idx[frame_idx])
        interacting_indices = object_points_idx[:, 1] != 0  
        interacting_body_indices = body_points_idx[interacting_indices]  
        
        # 获取对应的人体点坐标  
        # time_body = time.time()
        body_points = self.get_body_points(frame_idx)[interacting_body_indices]  
        # time_body2 = time.time()
        # print("get_body_points:", time_body2-time_body)
        
        # 获取对应的物体点坐标  
        object_points = self.get_object_points(frame_idx)

        obj_index = object_points_idx[interacting_indices][:, 0]  
        interactiong_obj = object_points[obj_index]

        
        # 创建对应点字典  
        corresponding_points = {  
            'body_points': body_points,  
            'object_points': interactiong_obj 
        }  
        
        return corresponding_points  

 
    
    def compute_contact_loss(self, frame_idx=None):
        """
        计算关键点接触损失，使用距离加权
        """
        if frame_idx is None:
            frame_idx = self.current_frame
        
        corresponding_points = self.get_corresponding_point(frame_idx)
        body_points = corresponding_points['body_points']
        object_points = corresponding_points['object_points']
        
        if (object_points.shape[0] == 0 | body_points.shape[0] == 0):
            return torch.tensor(0.0, device=body_points.device)
        distances = torch.norm(body_points - object_points, dim=1)
        weights = torch.pow(distances + 0.1, 2)  
        weights = weights / weights.sum()
        weighted_loss = torch.sum(weights * distances**2)
        
        return weighted_loss

    def compute_collision_loss(self, frame_idx=None, h_weight=10.0, threshold=None):  
        """  
        计算指定帧的碰撞损失  
        :param frame_idx: 帧索引，默认为当前帧  
        :param h_weight: 人体在物体内部的权重  
        :param threshold: 阈值  
        :return: 损失值  
        """  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        object_mesh = self.sampled_obj_meshes[frame_idx]
        # time_a = time.time()
        hverts = self.get_body_points(frame_idx, sampled=True).unsqueeze(0)  
        # time_b = time.time()
        # print("coll get body points:", time_b-time_a)
        overts = self.get_object_points(frame_idx, sampled=True).unsqueeze(0)  
        
        hfaces = self.get_body_faces(sampled=True)  
        ofaces = self.get_object_faces(frame_idx, sampled=True)

        hfaces = np.ascontiguousarray(hfaces, dtype=np.int64)
        ofaces = np.ascontiguousarray(ofaces, dtype=np.int64)
        
        hoi_dict = {  
            'smplx_v_centered': overts - torch.mean(overts, dim=1, keepdim=True),  
            'object_v_centered': hverts - torch.mean(overts, dim=1, keepdim=True),  
        }  
        h_in_o_collision_loss = HOCollisionLoss(ofaces).to(hverts.device)  
        h_in_o_loss_val = h_in_o_collision_loss(hoi_dict)  
        
        hoi_dict = {  
            'smplx_v_centered': hverts - torch.mean(hverts, dim=1, keepdim=True),  
            'object_v_centered': overts - torch.mean(hverts, dim=1, keepdim=True),   
        }  
        o_in_h_collision_loss = HOCollisionLoss(hfaces).to(hverts.device)  
        o_in_h_loss_val = o_in_h_collision_loss(hoi_dict)  
        
        out_loss = h_weight * h_in_o_loss_val + o_in_h_loss_val  
        
        if threshold is not None:  
            if out_loss < threshold:  
                return out_loss * 0.0  
        
        return out_loss  
    
    def compute_temporal_pose_smoothness(self, pose_weight=1.0, shape_weight=1.0, H=5, F=5, frame_idx=None):  
        """  
        计算人体姿态的时序平滑损失  
        :param lambda_pose: 平滑损失的权重  
        :return: 姿态平滑损失  
        """  
        if frame_idx is None:
            frame_idx = self.current_frame
        
        loss = torch.tensor(0.0, device=self.body_pose_params[0].device)
        if frame_idx == 0:
            return loss
        # 计算相邻帧之间的姿态变化  
        left = max(0, frame_idx - H)
        right = min(self.seq_length - 1, frame_idx + F)
        i = frame_idx
        pose_seq = torch.stack(self.body_pose_params[left:right], dim=0)
        shape_seq = torch.stack(self.shape_params[left:right], dim=0)
        pose_diff = pose_seq[1:] - pose_seq[:-1]
        shape_diff = shape_seq[1:] - shape_seq[:-1]
        loss = pose_weight*torch.sum(torch.norm(pose_diff, dim=1)**2) + shape_weight*torch.sum(torch.norm(shape_diff, dim=1)**2)  
        
        return loss
    def compute_temporal_object_smoothness(self, lambda_rot=1.0, lambda_trans=1.0, H=5, F=5, frame_idx=None):  
        """  
        计算物体运动的时序平滑损失  
        :param lambda_rot: 旋转平滑损失的权重  
        :param lambda_trans: 平移平滑损失的权重  
        :return: 物体平滑损失  
        """  
        if frame_idx is None:
            frame_idx = self.current_frame 
        all_loss = torch.tensor(0.0, device=self.obj_x_params[0].device)
        
        # 如果是静态物体，不需要计算时序平滑损失
        if self.is_static_object:
            return all_loss
            
        i = frame_idx
        left = max(0, i - H)
        right = min(self.seq_length - 1, i + F)
        # 只计算left~right区间的overts
        overts_list = []
        for idx in range(left, right):
            overts = torch.tensor(np.asarray(self.sampled_obj_meshes[idx].vertices), dtype=torch.float32).cuda()
            # overts = overts * self.obj_scale
            transforms = self.get_object_transform(idx).cuda().float()
            overts = torch.mm(overts, transforms).unsqueeze(0)
            overts = self.obj_transl_params[idx] + overts
            overts_list.append(overts.squeeze(0))
        overts_seq = torch.stack(overts_list, dim=0)
        verts_diff = overts_seq[1:] - overts_seq[:-1]
        all_loss += torch.sum(torch.norm(verts_diff, dim=1)**2)
        return all_loss 
    
        # if frame_idx is None:
        #     frame_idx = self.current_frame 
        
        # rot_loss = torch.tensor(0.0, device=self.obj_x_params[0].device)  
        # trans_loss = torch.tensor(0.0, device=self.obj_transl_params[0].device)  
        # if frame_idx == 0:
        #     return rot_loss
        # if frame_idx < H:
        #     H = frame_idx
        #     F = frame_idx
        # elif frame_idx >= self.seq_length - F:
        #     H = self.seq_length - F
        #     F = self.seq_length - F
        #     # 旋转角度变化  
        # i = frame_idx
        # x_diff = torch.stack(self.obj_x_params[i-H+1:i+F], dim=0) - torch.stack(self.obj_x_params[i-H:i+F-1], dim=0)  
        # y_diff = torch.stack(self.obj_y_params[i-H+1:i+F], dim=0) - torch.stack(self.obj_y_params[i-H:i+F-1], dim=0)  
        # z_diff = torch.stack(self.obj_z_params[i-H+1:i+F], dim=0) - torch.stack(self.obj_z_params[i-H:i+F-1], dim=0)  
        # trans_diff = torch.stack(self.obj_transl_params[i-H+1:i+F], dim=0) - torch.stack(self.obj_transl_params[i-H:i+F-1],dim=0)  
        # all_diff = torch.stack([x_diff, y_diff, z_diff], dim=0)
        # rot_loss = torch.sum(torch.norm(all_diff, dim=1)**2)
        # trans_loss = torch.sum(torch.norm(trans_diff, dim=1)**2)
        
        # return (lambda_rot * rot_loss + lambda_trans * trans_loss)
    def projected_2D_loss(self, frame_idx=None):
        """
        计算对应点投影到2D后的距离损失
        :param frame_idx: 帧索引，默认为当前帧
        :return: 2D投影距离损失
        """
        if frame_idx is None:
            frame_idx = self.current_frame
        
        # 获取对应点
        corresponding_points = self.get_corresponding_point(frame_idx)
        body_points = corresponding_points['body_points']
        object_points = corresponding_points['object_points']
        
        # 设置相机参数，参考compute_mask_loss中的相机设置
        image_size = (self.image_size, self.image_size)
        output = torch.load(self.video_dir + "/motion/result.pt")
        K = output["K_fullimg"][0].cuda()
        K_nf = K.clone()
        K_nf[0,0] = K[0, 0] * (image_size[0] / image_size[1])
        K_nf[0,2] = K[0, 2] * (image_size[0] / image_size[1])
        
        # 创建投影矩阵
        fx, fy = K_nf[0, 0], K_nf[1, 1]
        cx, cy = K_nf[0, 2], K_nf[1, 2]
        
        # 计算自动检测的3D对应点损失
        weighted_loss = torch.tensor(0.0, device='cuda')
    
        if body_points.shape[0] > 0 and object_points.shape[0] > 0:
            body_2d = torch.zeros((body_points.shape[0], 2), device=body_points.device)
            body_2d[:, 0] = fx * body_points[:, 0] / body_points[:, 2] + cx
            body_2d[:, 1] = fy * body_points[:, 1] / body_points[:, 2] + cy

            object_2d = torch.zeros((object_points.shape[0], 2), device=object_points.device)
            object_2d[:, 0] = fx * object_points[:, 0] / object_points[:, 2] + cx
            object_2d[:, 1] = fy * object_points[:, 1] / object_points[:, 2] + cy
            
            distances_2d = torch.norm(body_2d - object_2d, dim=1)

            weights = torch.pow(distances_2d + 0.1, 2)  # 加0.1避免距离为0时权重为0
            weights = weights / weights.sum()
            weighted_loss = torch.sum(weights * distances_2d**2)
        # 手动指定
        pairs_2d_loss = torch.tensor(0.0, device='cuda')
        
        if len(self.pairs_2d[frame_idx]) > 0:
            pairs = self.pairs_2d[frame_idx]
            
            # 获取物体的所有顶点
            object_vertices = self.get_object_points(frame_idx)
            
            # 收集所有的物体3D点索引和对应的2D点
            obj_indices = []
            target_2d_points = []
            
            for pair in pairs:
                obj_idx = pair[0]  # 物体点索引
                point_2d = pair[1]  # 2D坐标
                
                obj_indices.append(obj_idx)
                target_2d_points.append(point_2d)
            
            if len(obj_indices) > 0:
                # 获取指定索引的物体点
                selected_obj_points = object_vertices[obj_indices]
                
                # 将物体3D点投影到2D
                selected_obj_2d = torch.zeros((len(obj_indices), 2), device='cuda')
                selected_obj_2d[:, 0] = fx * selected_obj_points[:, 0] / selected_obj_points[:, 2] + cx
                selected_obj_2d[:, 1] = fy * selected_obj_points[:, 1] / selected_obj_points[:, 2] + cy
                
                # 将目标2D点转换为张量
                target_2d_tensor = torch.tensor(target_2d_points, dtype=torch.float32, device='cuda')
                
                # 计算2D距离
                pair_distances_2d = torch.norm(selected_obj_2d - target_2d_tensor, dim=1)
                
                # 使用相同的加权策略
                pair_weights = torch.pow(pair_distances_2d + 0.1, 2)
                if pair_weights.sum() > 0:
                    pair_weights = pair_weights / pair_weights.sum()
                
                # 计算加权损失
                pairs_2d_loss = torch.sum(pair_weights * pair_distances_2d**2)
        
        # 合并两部分损失
        total_loss = weighted_loss + 1.5 * pairs_2d_loss
        
        return total_loss

    # def compute_mask_loss(self, mask_weight=1.5, edge_weight=1e-3, frame_idx=None):
    #     if frame_idx is None:
    #         frame_idx = self.current_frame  
    #     image_size = (self.image_size//4, self.image_size//4)
    #     output = torch.load(self.video_dir + "/motion/result.pt")
    #     K = output["K_fullimg"][0].cuda()
    #     K_nf = K.clone()
    #     K_nf[0,0] = K[0, 0] * (image_size[0] / image_size[1])
    #     K_nf[0,2] = K[0, 2] * (image_size[0] / image_size[1])
    #     R = torch.eye(3, dtype=torch.float32).unsqueeze(0).cuda()
    #     T = torch.zeros(3, dtype=torch.float32).unsqueeze(0).cuda()
    #     i = frame_idx

    #     human_verts = self.get_body_points(i)
    #     # human_verts.retain_grad()
    #     human_faces = self.get_body_faces()
    #     human_faces = torch.tensor(human_faces, dtype=torch.int64).cuda()
    #     object_mesh = self.sampled_obj_meshes[i]  
    #     transform = self.get_object_transform(i).float()
    #     object_vertices = torch.tensor(np.asarray(object_mesh.vertices),  
    #                         dtype=torch.float32).cuda()  
    #     object_vertices = object_vertices*self.obj_scale
    #     object_vertices = torch.mm(object_vertices, transform)
    #     object_vertices = self.obj_transl_params[i] + object_vertices  
    #     object_faces = np.asarray(object_mesh.triangles)
    #     object_faces = torch.tensor(object_faces, dtype=torch.int64).cuda()  
    #     self.renderer = nr.renderer.Renderer(
    #         image_size=image_size[0],
    #         K=K_nf.unsqueeze(0),
    #         R=R,
    #         t=T,
    #         orig_size=self.image_size,
    #         anti_aliasing=True,
    #     )

    #     human_mask = self.renderer(human_verts.unsqueeze(0), human_faces.unsqueeze(0),  mode="silhouettes").squeeze()
    #     object_mask = self.renderer(object_vertices.unsqueeze(0), object_faces.unsqueeze(0),  mode="silhouettes").squeeze()
    #     # exit(0)
    #     # if (i==0):
    #     #     img = rgb_image.squeeze().detach().cpu().numpy()  # shape: [H, W]
    #     #     img = (img * 255).astype('uint8')
    #     #     img = Image.fromarray(img)
    #     #     img.save('image.png')
    #     # print(rgb_image.shape)
    #     # # 提取mask，使用sigmoid实现平滑的阈值操作
    #     # sigma = 1e6  # 控制sigmoid的陡峭程度
    #     # human_mask = torch.sigmoid(sigma * (rgb_image[..., 0] - 0.1))  # 红色通道表示人体
    #     # object_mask = torch.sigmoid(sigma * (rgb_image[..., 1] - 0.1))  # 绿色通道表示物体
    #     # print(human_mask.shape, object_mask.shape)
    #     # 调整大小
    #     human_mask = TF.resize(human_mask.unsqueeze(0), image_size, 
    #                         interpolation=TF.InterpolationMode.NEAREST)
    #     object_mask = TF.resize(object_mask.unsqueeze(0), image_size, 
    #                         interpolation=TF.InterpolationMode.NEAREST)
    #     """
    #     compute mask loss
    #     """
    #     if (i==0):
    #         mask = human_mask.squeeze().detach().cpu().numpy()  # shape: [H, W]
    #         mask = (mask * 255).astype('uint8')  # 转换为8位像素
    #         img = Image.fromarray(mask)
    #         img.save('resize_human_mask.png')
    #     if (i==0):
    #         mask = object_mask.squeeze().detach().cpu().numpy()  # shape: [H, W]
    #         mask = (mask * 255).astype('uint8')
    #         img = Image.fromarray(mask)
    #         img.save('resize_object_mask.png')
    #     gt_paths = {  
    #         'obj': os.path.join(self.video_dir, 'mask_dir', f"{str(i).zfill(5)}_mask.png"),  
    #         'human': os.path.join(self.video_dir, 'human_mask_dir', f"{str(i).zfill(5)}_mask.png")  
    #     }

    #     # frame_vis= os.path.join(self.video_dir, 'frame_list', f"{str(i).zfill(5)}.jpg")
    #     # frame_vis_p= Image.open(frame_vis).convert("RGB")
    #     # ## overlay rgb_image to frame_vis_p and save
    #     # frame_vis_p = np.array(frame_vis_p)
    #     # frame_vis_p=process_frame2square(frame_vis_p)
    #     # frame_vis_p=cv2.resize(frame_vis_p, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    #     # rgb_image= cv2.resize(rgb_image.squeeze().detach().cpu().numpy() , (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)*255
    #     # mask = human_mask.squeeze().detach().cpu().numpy().reshape(image_size[0], image_size[1],1)
    #     # # print("frame_vis_p.shape:", frame_vis_p.shape, "mask.shape:", mask.shape, "rgb_image.shape:", rgb_image.shape)
    #     # frame_vis_p = (frame_vis_p * (1 - mask) + rgb_image* mask).astype('uint8')
    #     # img = Image.fromarray(frame_vis_p)
    #     # img.save('overlay_frame.png')


    #     render_masks = {  
    #         'obj': object_mask,
    #         'human': human_mask
    #     }  
    #     if not hasattr(self, 'cached_gt_masks') or i not in self.cached_gt_masks:  
    #         if not hasattr(self, 'cached_gt_masks'):  
    #             self.cached_gt_masks = {}  
    #         self.cached_gt_masks[i] = {}  
    #         for key, path in gt_paths.items():  
    #             mask = Image.open(path).convert("L")
    #             gt_mask = process_frame2square_mask(np.array(mask, copy=True)[:,:,np.newaxis])
    #             gt_mask = torch.from_numpy(gt_mask).float().div(255).to(render_masks[key].device)[:,:,0]
    #             downsampled_gt = F.interpolate(  
    #                 gt_mask.unsqueeze(0).unsqueeze(0),  
    #                 size=render_masks[key].squeeze().shape,  
    #                 mode='area'  
    #             ).squeeze()  
    #             self.cached_gt_masks[i][key] = downsampled_gt  
    #     # time_end = time.time()
    #     # print("get gt mask time:", time_end-time_get)
        
    #     time_compute = time.time()
    #     h_gt_mask = self.cached_gt_masks[i]['human'].unsqueeze(0)
    #     o_gt_mask = self.cached_gt_masks[i]['obj'].unsqueeze(0)
    #     h_render_mask = render_masks['human']*(1 - o_gt_mask)
    #     o_render_mask = render_masks['obj']*(1 - h_gt_mask)
    #     power=0.5
    #     batch_size=1

    #     ## mask loss:
    #     # print("mask loss shape:", h_render_mask.shape, o_render_mask.shape, h_gt_mask.shape, o_gt_mask.shape)
    #     h_mask_loss= F.mse_loss(h_render_mask, h_gt_mask)
    #     o_mask_loss= F.mse_loss(o_render_mask, o_gt_mask)

    #     kernel_size = 7
    #     pool = torch.nn.MaxPool2d(
    #         kernel_size=kernel_size, stride=1, padding=(kernel_size // 2)
    #     )
    #     h_edge_render=pool(h_render_mask)-h_render_mask
    #     o_edge_render=pool(o_render_mask)-o_render_mask
    #     h_edge_gt=pool(h_gt_mask)-h_gt_mask
    #     o_edge_gt=pool(o_gt_mask)-o_gt_mask
    #     h_edt = distance_transform_edt(1 - (h_edge_gt.detach().cpu().numpy() > 0)) ** (power * 2)
    #     o_edt = distance_transform_edt(1 - (o_edge_gt.detach().cpu().numpy() > 0)) ** (power * 2)
    #     # print("h_edt_max", h_edt.max(), "o_edt_max:", o_edt.max())
    #     h_edt = torch.from_numpy(h_edt).repeat(batch_size, 1, 1).float().cuda()
    #     o_edt = torch.from_numpy(o_edt).repeat(batch_size, 1, 1).float().cuda()

    #     edge_img = o_edge_render.squeeze(0).detach().cpu().numpy()
    #     edge_img_gt = o_edge_gt.squeeze(0).detach().cpu().numpy()
    #     cv2.imwrite('./o_mask.png',o_render_mask.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./o_mask_gt.png',o_gt_mask.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./h_mask.png',h_render_mask.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./h_mask_gt.png',h_gt_mask.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./h_edge.png',h_edge_render.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./h_edge_gt.png',h_edge_gt.squeeze(0).detach().cpu().numpy()*255)
    #     cv2.imwrite('./o_edge.png',edge_img*255)
    #     cv2.imwrite('./o_edge_gt.png',edge_img_gt*255)
    #     h_edge_loss=torch.sum(h_edge_render * h_edt, dim=(1, 2))
    #     o_edge_loss=torch.sum(o_edge_render * o_edt, dim=(1, 2))
    #     # print("h_edge_loss:", h_edge_loss.item(), "o_edge_loss:", o_edge_loss.item())
    #     # print("h_mask_loss:", h_mask_loss.item(), "o_mask_loss:", o_mask_loss.item())
    #     total_loss = 0.5*(edge_weight * h_edge_loss + mask_weight * h_mask_loss)+ 1.5*(edge_weight * o_edge_loss + mask_weight * o_mask_loss)
    #     # total_loss = mask_weight * h_mask_loss+ mask_weight * o_mask_loss
    #     return total_loss
    def _interpolate_frames(self, interval):
        """
        对未优化的帧进行线性插值
        :param interval: 优化间隔
        """
        with torch.no_grad():
            for i in range(0, self.seq_length - interval, interval):
                start_frame = i
                end_frame = i + interval
                # 对中间帧进行插值
                for mid_frame in range(start_frame + 1, end_frame):
                    alpha = (mid_frame - start_frame) / interval
                    self.body_pose_params[mid_frame].copy_(
                        (1 - alpha) * self.body_pose_params[start_frame] + 
                        alpha * self.body_pose_params[end_frame]
                    )
                    self.shape_params[mid_frame].copy_(
                        (1 - alpha) * self.shape_params[start_frame] + 
                        alpha * self.shape_params[end_frame]
                    )
                    self.left_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.left_hand_params[start_frame] + 
                        alpha * self.left_hand_params[end_frame]
                    )
                    self.right_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.right_hand_params[start_frame] + 
                        alpha * self.right_hand_params[end_frame]
                    )
                    self.obj_x_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_x_params[start_frame] + 
                        alpha * self.obj_x_params[end_frame]
                    )
                    
                    self.obj_y_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_y_params[start_frame] + 
                        alpha * self.obj_y_params[end_frame]
                    )
                    
                    self.obj_z_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_z_params[start_frame] + 
                        alpha * self.obj_z_params[end_frame]
                    )

                    self.obj_transl_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_transl_params[start_frame] + 
                        alpha * self.obj_transl_params[end_frame]
                    )
            
            # 处理最后一段
            last_optimized = (self.seq_length - 1) // interval * interval
            if last_optimized != self.seq_length - 1:
                start_frame = last_optimized
                end_frame = self.seq_length - 1
                
                for mid_frame in range(start_frame + 1, end_frame):
                    alpha = (mid_frame - start_frame) / (end_frame - start_frame)
                    self.body_pose_params[mid_frame].copy_(
                        (1 - alpha) * self.body_pose_params[start_frame] + 
                        alpha * self.body_pose_params[end_frame]
                    )
                    
                    self.shape_params[mid_frame].copy_(
                        (1 - alpha) * self.shape_params[start_frame] + 
                        alpha * self.shape_params[end_frame]
                    )
                    
                    self.left_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.left_hand_params[start_frame] + 
                        alpha * self.left_hand_params[end_frame]
                    )
                    
                    self.right_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.right_hand_params[start_frame] + 
                        alpha * self.right_hand_params[end_frame]
                    )
                    self.obj_x_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_x_params[start_frame] + 
                        alpha * self.obj_x_params[end_frame]
                    )
                    
                    self.obj_y_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_y_params[start_frame] + 
                        alpha * self.obj_y_params[end_frame]
                    )
                    
                    self.obj_z_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_z_params[start_frame] + 
                        alpha * self.obj_z_params[end_frame]
                    )
                    
                    self.obj_transl_params[mid_frame].copy_(
                        (1 - alpha) * self.obj_transl_params[start_frame] + 
                        alpha * self.obj_transl_params[end_frame]
                    )    

 
    def optimize(self,   
                steps=100,   
                print_every=10,   
                contact_weight=80,   
                collision_weight=5, 
                mask_weight=0.05,
                project_2d_weight=3.5e-3,
                optimize_per_frame=True,
                optimize_interval=8):
        """  
        执行优化  
        :param steps: 优化步数  
        :param print_every: 每隔多少步打印一次损失  
        :param contact_weight: 接触损失权重  
        :param collision_weight: 碰撞损失权重  
        :param temporal_pose_weight: 姿态时序平滑权重  
        :param temporal_object_weight: 物体时序平滑权重  
        :param optimize_per_frame: 是否逐帧优化，默认同时优化整个序列  
        :param optimize_interval: 优化间隔，每隔多少帧优化一次，默认为1（即每帧都优化）
        """  
        # time0 = time.time()
        self.training_setup()  
        if optimize_per_frame:  
            # 逐帧优化，但每隔 optimize_interval 帧优化一次
            frames_to_optimize = list(range(0, self.seq_length, optimize_interval))
            if frames_to_optimize[-1] != self.seq_length:
                frames_to_optimize.append(self.seq_length-1)  # 确保最后一帧被优化
            for frame_idx in tqdm(frames_to_optimize):
                self.current_frame = frame_idx
                for step in range(steps): 
                    # print("step:", step) 
                    # human_mesh = o3d.geometry.TriangleMesh()
                    # human_mesh.vertices = o3d.utility.Vector3dVector(self.get_body_points(frame_idx).detach().cpu().numpy())
                    # human_mesh.triangles = o3d.utility.Vector3iVector(np.array(self.get_body_faces(), dtype=np.int32))
                    # o3d.io.write_triangle_mesh(os.path.join(self.video_dir, f"human_mesh_{frame_idx}.obj"), human_mesh)
                    # o3d.io.write_triangle_mesh(os.path.join(self.video_dir, f"object_mesh_{frame_idx}.obj"), self.object_meshes[frame_idx])
                        
                    self.optimizer.zero_grad()
                    # time0 = time.time()
                    contact_loss = self.compute_contact_loss(frame_idx)
                    # projected_2d_loss = self.projected_2D_loss(frame_idx)
                    # mask_loss= self.compute_mask_loss(5e3, 1e-5, frame_idx)
                    collision_loss = self.compute_collision_loss(frame_idx, h_weight=10.0, threshold=0) + 1e-5
                    loss = (contact_weight * contact_loss
                            # + project_2d_weight * projected_2d_loss
                            + collision_weight * collision_loss
                            )
                    # loss = mask_weight * mask_loss
                    # loss = contact_weight * contact_loss + project_2d_weight * projected_2d_loss
                    param_idx = 0 if self.is_static_object else frame_idx
                    if torch.any(torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit):
                        limit_mask = torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit
                        total_loss = loss + 1e6*F.mse_loss(self.obj_transl_params[param_idx][limit_mask], self.obj_transl_limit[limit_mask])
                    else:
                        total_loss = loss
                    total_loss.backward()
                    # with torch.no_grad():
                    #     joint_weights = self.compute_joint_keypoint_dist(frame_idx).to(self.body_pose_params[frame_idx].device)
                    # for j in range(24):
                    #     slice_idx = slice(j*3, (j+1)*3)
                    # self.body_pose_params[frame_idx].grad[slice_idx] *= joint_weights[j]
                    # 仅更新掩码中为True的参数  

                    with torch.no_grad():
                        # if frame_idx != 0:
                        #     self.obj_scale.grad[...] = 0
                        if self.is_static_object and frame_idx != 0:
                            if self.obj_transl_params[frame_idx].grad is not None:
                                self.obj_transl_params[frame_idx].grad[...] = 0
                            if self.obj_x_params[frame_idx].grad is not None:
                                self.obj_x_params[frame_idx].grad[...] = 0
                            if self.obj_y_params[frame_idx].grad is not None:
                                self.obj_y_params[frame_idx].grad[...] = 0
                            if self.obj_z_params[frame_idx].grad is not None:
                                self.obj_z_params[frame_idx].grad[...] = 0
                        if self.body_mask is not None:
                            with torch.no_grad():
                                self.body_pose_params[frame_idx].grad[~self.body_mask] = 0
                        if self.left_hand_mask is not None:
                            with torch.no_grad():
                                self.left_hand_params[frame_idx].grad[~self.left_hand_mask] = 0
                        if self.right_hand_mask is not None:
                            with torch.no_grad():
                                self.right_hand_params[frame_idx].grad[~self.right_hand_mask] = 0
                    self.optimizer.step()
                    # self.scheduler.step()
                    # time6 = time.time()
                    if step % print_every == 0:
                        tqdm.write(f"Frame {frame_idx}, Step {step}: Loss = {loss.item():.4f}, "
                              f"Contact = {contact_loss.item():.4f}, "
                              # f"Projected_2D = {projected_2d_loss.item():.4f}, "
                              f"Collision = {collision_loss.item():.4f}"
                                   )
            
            if self.is_static_object:
                with torch.no_grad():
                    first_transl_value = self.obj_transl_params[0].clone()
                    first_x_value = self.obj_x_params[0].clone()
                    first_y_value = self.obj_y_params[0].clone()
                    first_z_value = self.obj_z_params[0].clone()
                    
                    for frame_idx in range(1, self.seq_length):
                        self.obj_transl_params[frame_idx].data.copy_(first_transl_value)
                        self.obj_x_params[frame_idx].data.copy_(first_x_value)
                        self.obj_y_params[frame_idx].data.copy_(first_y_value)
                        self.obj_z_params[frame_idx].data.copy_(first_z_value)

            if not self.is_static_object and optimize_interval > 1 :
                self._interpolate_frames(optimize_interval)
            # self.save_sequence(os.path.join(self.video_dir, "optimized_sequence"))
            time2 = time.time()
            # print(f"optimize time: {time2 - time0}")
            # self._post_process_smoothing(smooth_steps=30, smooth_lr=0.001)
            self._kalman_smoothing()
            # self._post_process_smoothing(smooth_steps=20, smooth_lr=0.001)
    
    def _kalman_smoothing(self):
        def apply_kalman(param_list, observation_weight=5.0, transition_cov=0.1, param_name=""):
            num_frames = len(param_list)
            if num_frames == 0: return
            param_shape = param_list[0].shape
            num_dims = param_list[0].numel()
            
            with torch.no_grad():
                observations = np.zeros((num_frames, num_dims), dtype=np.float32)
                for frame_idx in range(num_frames):
                    flat_param = param_list[frame_idx].view(-1).cpu().numpy()
                    observations[frame_idx] = flat_param
                
                data_variance = np.var(observations, axis=0)
                data_variance = np.maximum(data_variance, 1e-6)

                kf = KalmanFilter(
                    initial_state_mean=observations[0],
                    initial_state_covariance=np.diag(data_variance),
                    transition_matrices=np.eye(num_dims),
                    observation_matrices=np.eye(num_dims),
                    observation_covariance=np.diag(data_variance) * observation_weight,
                    transition_covariance=np.diag(data_variance) * transition_cov,
                )

                smooth_states, _ = kf.smooth(observations)
                device = param_list[0].device
                for frame_idx in range(num_frames):
                    smoothed_flat = torch.from_numpy(smooth_states[frame_idx]).float().to(device)
                    param_list[frame_idx].data = smoothed_flat.view(param_shape)
        transl = [nn.Parameter(p) for p in self.transl]
        apply_kalman(transl, observation_weight=1, transition_cov=0.1, param_name="transl")
        self.transl = [p.data for p in transl]
        apply_kalman(self.body_pose_params, observation_weight=0.05, transition_cov=0.1, param_name="body_pose")
        apply_kalman(self.shape_params, observation_weight=0.1, transition_cov=0.1, param_name="shape")
        apply_kalman(self.left_hand_params, observation_weight=0.1, transition_cov=0.1, param_name="left_hand")
        apply_kalman(self.right_hand_params, observation_weight=0.1, transition_cov=0.1, param_name="right_hand")
 
        if not self.is_static_object:
            # Smoothing via Lie Algebra (se3) using GTSAM for robustness.
            with torch.no_grad():
                pose_seq_gtsam = []
                # Move tensors to CPU and convert to numpy for gtsam
                final_R_list_cpu = [self.get_object_transform(i).cpu().numpy() for i in range(self.seq_length)]
                t_base_list_cpu = [t.cpu().numpy() for t in self.base_obj_transl]
                t_residual_list_cpu = [t.cpu().numpy() for t in self.obj_transl_params]
                centers_cpu = np.array(self.centers)

                for i in range(self.seq_length):
                    R_base = self.base_obj_R[i].cpu().numpy()
                    R_residual = final_R_list_cpu[i]
                    R_final_no_initial = R_residual @ R_base
                    R_final = R_final_no_initial @ self.initial_R[i]

                    t_base = t_base_list_cpu[i]
                    t_residual = t_residual_list_cpu[i]
                    
                    center = centers_cpu[i]
                    effective_translation = R_final_no_initial @ center + R_residual @ t_base + t_residual

                    pose_gtsam = gtsam.Pose3(gtsam.Rot3(R_final), effective_translation)
                    pose_seq_gtsam.append(pose_gtsam)
                twists = []
                for i in range(1, self.seq_length):
                    pose_i_minus_1 = pose_seq_gtsam[i-1]
                    pose_i = pose_seq_gtsam[i]
                    relative_pose = pose_i_minus_1.inverse().compose(pose_i)
                    twist = gtsam.Pose3.Logmap(relative_pose) # 6D se3 vector
                    twists.append(nn.Parameter(torch.from_numpy(twist).float()))
                
                if not twists: 
                    return
                apply_kalman(twists, observation_weight=3, transition_cov=0.1, param_name="twists")
                smoothed_poses_gtsam = [pose_seq_gtsam[0]]
                for i in range(len(twists)):
                    smoothed_twist_np = twists[i].data.numpy()
                    relative_pose_smoothed = gtsam.Pose3.Expmap(smoothed_twist_np)
                    new_pose = smoothed_poses_gtsam[i].compose(relative_pose_smoothed)
                    smoothed_poses_gtsam.append(new_pose)

                smoothed_final_R_list = [p.rotation().matrix() for p in smoothed_poses_gtsam]
                smoothed_effective_transl_list = [p.translation() for p in smoothed_poses_gtsam]
                
                device = self.obj_x_params[0].device
                smoothed_final_R_stack = torch.from_numpy(np.array(smoothed_final_R_list)).float().to(device)
                smoothed_effective_transl_list = [p.translation() for p in smoothed_poses_gtsam]
                
                device = self.obj_x_params[0].device
                smoothed_final_R_stack = torch.from_numpy(np.array(smoothed_final_R_list)).float().to(device)
                smoothed_effective_transl = torch.from_numpy(np.array(smoothed_effective_transl_list)).float().to(device)

                base_R_inv_stack = torch.stack(self.base_obj_R).transpose(1, 2)
                initial_R_stack = torch.from_numpy(np.array(self.initial_R)).float().to(device)
                initial_R_inv_tensor = initial_R_stack.transpose(1, 2)
                
                smoothed_final_R_no_initial_stack = torch.matmul(smoothed_final_R_stack, initial_R_inv_tensor)
                new_residual_R_stack = torch.matmul(smoothed_final_R_no_initial_stack, base_R_inv_stack)
                new_residual_euler_rad = matrix_to_euler_angles(new_residual_R_stack, "ZYX")
                new_residual_euler_deg = torch.rad2deg(new_residual_euler_rad)

                self.obj_z_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 0]]
                self.obj_y_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 1]]
                self.obj_x_params = [nn.Parameter(p) for p in new_residual_euler_deg[:, 2]]

                centers_stack = torch.tensor(self.centers, device=device, dtype=torch.float32)
                base_t_stack = torch.stack(self.base_obj_transl)

                rotated_centers = torch.bmm(smoothed_final_R_no_initial_stack, centers_stack.unsqueeze(-1)).squeeze(-1)
                new_t_final = smoothed_effective_transl - rotated_centers
                
                rotated_base_t = torch.bmm(new_residual_R_stack, base_t_stack.unsqueeze(-1)).squeeze(-1)
                new_residual_transl = new_t_final - rotated_base_t
                self.obj_transl_params = [nn.Parameter(p) for p in new_residual_transl]
                # smoothed_R_stack = torch.from_numpy(np.array(smoothed_final_R_list)).float().to(device)
                # smoothed_t_stack = torch.from_numpy(np.array(smoothed_effective_transl_list)).float().to(device)

                # # Instead of decomposing, we store the final smoothed trajectory directly.
                # self.smoothed_R_seq = [R for R in smoothed_R_stack]
                # self.smoothed_t_seq = [t for t in smoothed_t_stack]
 
    
    def save_sequence(self, output_dir):  
        """  
        保存优化后的序列模型到指定目录  
        :param output_dir: 输出目录  
        """  
        os.makedirs(output_dir, exist_ok=True)  
        
        for i in range(self.seq_length):
            frame_dir = os.path.join(output_dir, f'frame_{i + self.start_frame:04d}')  
            os.makedirs(frame_dir, exist_ok=True)   
            self.current_frame = i 
            human_faces = self.get_body_faces(sampled=False)  
            human_verts = self.get_body_points(i, sampled=False).detach().cpu().numpy()
            object_vertices = self.get_object_points(i, sampled=True).detach().cpu().numpy()
            incam_params = (self.global_orient[i], self.transl[i])
            global_params = (self.global_body_params["global_orient"][i], self.global_body_params["transl"][i])
            human_verts, object_vertices = transform_to_global(human_verts, object_vertices, incam_params, global_params)
            # 保存人体mesh  
            h_mesh = o3d.geometry.TriangleMesh()  
            h_mesh.vertices = o3d.utility.Vector3dVector(human_verts)  
            h_mesh.triangles = o3d.utility.Vector3iVector(human_faces)  
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'human.obj'), h_mesh)  
            
            # 保存物体mesh  
            obj_mesh = o3d.geometry.TriangleMesh()  
            obj_mesh.vertices = o3d.utility.Vector3dVector(object_vertices)  
            obj_mesh.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=True))  
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'object.obj'), obj_mesh)  
            
            # 关键点连线  
            corresponding_points = self.get_corresponding_point(i)  
            body_points = corresponding_points['body_points'].detach().cpu().numpy()  
            object_points = corresponding_points['object_points'].detach().cpu().numpy()  
            
            lines = [[i, i + len(body_points)] for i in range(len(body_points))]  
            points = np.vstack((body_points, object_points))  
            colors = [[0, 1, 0] for _ in range(len(lines))]  
            
            line_set = o3d.geometry.LineSet(  
                points=o3d.utility.Vector3dVector(points),  
                lines=o3d.utility.Vector2iVector(lines),  
            )  
            line_set.colors = o3d.utility.Vector3dVector(colors)  
            o3d.io.write_line_set(os.path.join(frame_dir, 'contact_points.ply'), line_set)  
    
    # def create_visualization_video(self, output_dir, K, R, T, video_path=None, fps=3,
    #                                         rotation_angle=0, rotation_axis='y'):
    #
    #     def rotate_camera(rotation_angle, rotation_axis):
    #         camera_rotation_rad = math.radians(rotation_angle)
    #         if rotation_axis.lower() == 'y':
    #             camera_self_rotation = np.array([
    #                 [math.cos(camera_rotation_rad), 0, math.sin(camera_rotation_rad)],
    #                 [0, 1, 0],
    #                 [-math.sin(camera_rotation_rad), 0, math.cos(camera_rotation_rad)]
    #             ])
    #         elif rotation_axis.lower() == 'x':
    #             camera_self_rotation = np.array([
    #                 [1, 0, 0],
    #                 [0, math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad)],
    #                 [0, math.sin(camera_rotation_rad), math.cos(camera_rotation_rad)]
    #             ])
    #         elif rotation_axis.lower() == 'z':
    #             camera_self_rotation = np.array([
    #                 [math.cos(camera_rotation_rad), -math.sin(camera_rotation_rad), 0],
    #                 [math.sin(camera_rotation_rad), math.cos(camera_rotation_rad), 0],
    #                 [0, 0, 1]
    #             ])
    #         return camera_self_rotation
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     # ===== 渲染设置 =====
    #     render_size = self.image_size // 3
    #     render = o3d.visualization.rendering.OffscreenRenderer(render_size, render_size)
    #     render.scene.set_background([1.0, 1.0, 1.0, 1.0])
    #     human_mat = o3d.visualization.rendering.MaterialRecord()
    #     human_mat.base_color = [0.7, 0.3, 0.3, 1.0]
    #     human_mat.shader = "defaultLit"
    #
    #     object_mat = o3d.visualization.rendering.MaterialRecord()
    #     object_mat.base_color = [0.3, 0.5, 0.7, 1.0]
    #     object_mat.shader = "defaultLit"
    #
    #     middle_frame = self.seq_length // 2
    #     self.current_frame = middle_frame
    #     object_vertices = self.get_object_points(middle_frame, sampled=True).detach().cpu().numpy()
    #     human_vertices = self.get_body_points(middle_frame, sampled=False).detach().cpu().numpy()
    #     incam_params = (self.global_orient[middle_frame], self.transl[middle_frame])
    #     global_params = (self.global_body_params["global_orient"][middle_frame], self.global_body_params["transl"][middle_frame])
    #     human_vertices, object_vertices = transform_to_global(human_vertices, object_vertices, incam_params, global_params)
    #     all_vertices = np.vstack((human_vertices, object_vertices))
    #     camera_params = create_camera_for_object(
    #         all_vertices,
    #         image_width=render_size,
    #         image_height=render_size,
    #         distance_factor=4.0,
    #         fov_degrees=60.0  # 减小FOV来"放大"主体
    #     )
    #     K = camera_params['intrinsics']
    #     R = camera_params['rotation_matrix']
    #     T = camera_params['camera_position']
    #     scene_center = np.mean(all_vertices, axis=0).reshape(3, 1)
    #
    #     R_np = R.cpu().numpy() if torch.is_tensor(R) else np.array(R)
    #     T_np = T.cpu().numpy() if torch.is_tensor(T) else np.array(T)
    #     if T_np.ndim == 1:
    #         T_np = T_np.reshape(3, 1)
    #     original_camera_pos = T_np
    #
    #     camera_self_rotation = rotate_camera(-rotation_angle, rotation_axis)
    #     rotated_R = R_np @ camera_self_rotation
    #
    #     world_rotation_matrix = rotate_camera(rotation_angle, rotation_axis)
    #     camera_relative = original_camera_pos - scene_center
    #     rotated_camera_relative = world_rotation_matrix @ camera_relative
    #     new_camera_pos = rotated_camera_relative + scene_center
    #     final_R = rotated_R
    #     final_T = new_camera_pos.flatten()
    #     # flip_rotation = rotate_camera(180, 'z')
    #     # final_R = final_R @ flip_rotation
    #
    #     extrinsic_matrix = np.eye(4)
    #     extrinsic_matrix[:3, :3] = final_R
    #     extrinsic_matrix[:3, 3] = final_T
    #     view_matrix = np.linalg.inv(extrinsic_matrix)
    #     def create_camera_pyramid(length=0.03, base=0.02):
    #         # 顶点数据：锥体的尖端+底面4点
    #         tip = np.array([[0, 0, length]])
    #         half = base / 2
    #         base_pts = np.array([
    #             [ half,  half, 0],
    #             [-half,  half, 0],
    #             [-half, -half, 0],
    #             [ half, -half, 0]
    #         ])
    #         vertices = np.vstack([tip, base_pts])
    #         triangles = np.array([
    #             [0, 1, 2],
    #             [0, 2, 3],
    #             [0, 3, 4],
    #             [0, 4, 1],
    #             [1, 2, 3],
    #             [1, 3, 4]
    #         ])
    #         mesh = o3d.geometry.TriangleMesh()
    #         mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #         mesh.triangles = o3d.utility.Vector3iVector(triangles)
    #         mesh.compute_vertex_normals()
    #         return mesh
    #
    #     def apply_transform(mesh, R, t):
    #         mesh.rotate(R, center=np.zeros(3))
    #         mesh.translate(t)
    #         return mesh
    #
    #     camera_pyramid_length = np.linalg.norm(scene_center) * 0.12
    #     camera_pyramid_base = camera_pyramid_length * 0.66
    #
    #     orig_pyramid = create_camera_pyramid(camera_pyramid_length, camera_pyramid_base)
    #     orig_pyramid.paint_uniform_color([1.0, 0.2, 0.2])  # 红色
    #     orig_pyramid = apply_transform(orig_pyramid, R_np, original_camera_pos.flatten())
    #     o3d.io.write_triangle_mesh(os.path.join(output_dir, "camera_original.obj"), orig_pyramid)
    #
    #     rotated_pyramid = create_camera_pyramid(camera_pyramid_length, camera_pyramid_base)
    #     rotated_pyramid.paint_uniform_color([0.2, 0.4, 1.0])  # 蓝色
    #     rotated_pyramid = apply_transform(rotated_pyramid, final_R, new_camera_pos.flatten())
    #     o3d.io.write_triangle_mesh(os.path.join(output_dir, "camera_rotated.obj"), rotated_pyramid)
    #     # End of 相机可视化
    #
    #     K_np = K.cpu().numpy() if torch.is_tensor(K) else np.array(K)
    #     camera = o3d.camera.PinholeCameraIntrinsic(
    #         width=render_size,
    #         height=render_size,
    #         fx=K_np[0, 0],
    #         fy=K_np[1, 1],
    #         cx=K_np[0, 2],
    #         cy=K_np[1, 2]
    #     )
    #     # original_distance = np.linalg.norm(original_camera_pos - object_center)
    #     # new_distance = np.linalg.norm(new_camera_pos - object_center)
    #     # print(f"原始相机到物体距离: {original_distance:.3f}")
    #     # print(f"新相机到物体距离: {new_distance:.3f}")
    #     # print(f"距离差异: {abs(original_distance - new_distance):.6f}")
    #     # camera_forward = -final_R[2, :]
    #     # to_object = (object_center.flatten() - new_camera_pos.flatten())
    #     # to_object = to_object / np.linalg.norm(to_object)
    #
    #     # print(f"相机前向量: {camera_forward}")
    #     # print(f"相机到物体方向: {to_object}")
    #     # print(f"方向一致性 (dot product): {np.dot(camera_forward, to_object):.3f}")
    #     os.makedirs(output_dir, exist_ok=True)
    #     human_faces = np.array(self.get_body_faces(sampled=False), dtype=np.int32)
    #
    #
    #     for i in tqdm(range(0, self.seq_length, 2), desc=f"Processing frames with {rotation_angle}° rotation"):
    #
    #         human_verts = self.get_body_points(i, sampled=False).detach().cpu().numpy()
    #         object_vertices = self.get_object_points(i, sampled=True).detach().cpu().numpy()
    #         incam_params = (self.global_orient[i], self.transl[i])
    #         global_params = (self.global_body_params["global_orient"][i], self.global_body_params["transl"][i])
    #         human_verts, object_vertices = transform_to_global(human_verts, object_vertices, incam_params, global_params)
    #
    #         # transform to global
    #
    #
    #         render.scene.clear_geometry()
    #         human_mesh = o3d.geometry.TriangleMesh()
    #         human_mesh.vertices = o3d.utility.Vector3dVector(human_verts)
    #         human_mesh.triangles = o3d.utility.Vector3iVector(human_faces)
    #         human_mesh.compute_vertex_normals()
    #
    #         transformed_object_mesh = o3d.geometry.TriangleMesh()
    #         transformed_object_mesh.vertices = o3d.utility.Vector3dVector(object_vertices)
    #         transformed_object_mesh.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=True))
    #         transformed_object_mesh.compute_vertex_normals()
    #         render.scene.add_geometry("human", human_mesh, human_mat)
    #         render.scene.add_geometry("object", transformed_object_mesh, object_mat)
    #
    #         render.setup_camera(camera, view_matrix)
    #         rendered_img = render.render_to_image()
    #         start_frame = (self.start_frame // 2) * 2
    #         frame_path = os.path.join(output_dir, f"frame_{i+start_frame:04d}.png")
    #         o3d.io.write_image(frame_path, rendered_img)

    def create_visualization_video(self, output_dir, K, video_path=None, fps=3, clear=True):
        if os.path.exists(output_dir) and clear:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        human_faces = np.array(self.get_body_faces(sampled=True), dtype=np.int32)
        obj_faces= np.array(self.sampled_obj_meshes[0].triangles, dtype=np.int32)
        renderer = Renderer(self.width, self.height, device="cuda",faces_human=human_faces,faces_obj=obj_faces,K=K)

        for i in tqdm(range(0, self.seq_length, 2), desc="rendering frames"):

            human_verts = self.get_body_points(i, sampled=True)
            object_mesh = self.sampled_obj_meshes[i]
            transform = self.get_object_transform(i).detach().cpu().numpy()
            if i==0:
                R_final, t_final = self.get_object_params(i)
                print('R_final',R_final,'t_final',t_final,'transl',self.transl[i],'global_orient',self.global_orient[i])
                # body_param= self.get_body_params(i)
                # print('transform',transform,'transl',self.transl[i])

            object_vertices = self.get_object_points(i,sampled=True)
            # object_vertices=torch.from_numpy(object_vertices).float().cuda()
            img_raw = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            ## use object vertex color as object color, object_mesh.vertex_color
            if hasattr(object_mesh, 'vertex_colors'):
                object_color = np.asarray(object_mesh.vertex_colors)
                print("object_color shape:", object_color.shape)
            else:
                object_color = [0.3, 0.5, 0.7]

            img = renderer.render_mesh_hoi(human_verts, object_vertices, img_raw, [0.8, 0.8, 0.8],object_color)


            start_frame = (self.start_frame // 2) * 2
            frame_path = os.path.join(output_dir, f"frame_{i + start_frame:04d}.png")

            img = Image.fromarray(img)
            img.save(frame_path)


    def create_visualization_video_front(self, output_dir, K, R, T, video_path=None, fps=3):  
        # time0 = time.time()
        os.makedirs(output_dir, exist_ok=True)  
        render_size = self.image_size // 3 
        render = o3d.visualization.rendering.OffscreenRenderer(render_size, render_size)   
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # 白色背景
        human_mat = o3d.visualization.rendering.MaterialRecord()  
        human_mat.base_color = [0.7, 0.3, 0.3, 1.0]
        human_mat.shader = "defaultLit"  
        object_mat = o3d.visualization.rendering.MaterialRecord()  
        object_mat.base_color = [0.3, 0.5, 0.7, 1.0]    
        object_mat.shader = "defaultLit"  
        K_np = K.cpu().numpy() if torch.is_tensor(K) else np.array(K)  
        camera = o3d.camera.PinholeCameraIntrinsic(  
            width=render_size,  
            height=render_size,  
            fx=K_np[0, 0] * (render_size / self.image_size),  # 按比例缩放
            fy=K_np[1, 1] * (render_size / self.image_size),  
            cx=K_np[0, 2] * (render_size / self.image_size),  
            cy=K_np[1, 2] * (render_size / self.image_size)  
        )  
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R.cpu().numpy()
        extrinsic_matrix[:3, 3] = T.cpu().numpy()
        view_matrix = np.linalg.inv(extrinsic_matrix)
        human_faces = np.array(self.get_body_faces(sampled=True), dtype=np.int32) 
        for i in tqdm(range(0, self.seq_length, 2), desc="rendering frames"):  
            render.scene.clear_geometry()  
            self.current_frame = min(i, self.seq_length-1)  
            human_verts = self.get_body_points(i, sampled=True).detach().cpu().numpy()  
            human_mesh = o3d.geometry.TriangleMesh()  
            human_mesh.vertices = o3d.utility.Vector3dVector(human_verts)  
            human_mesh.triangles = o3d.utility.Vector3iVector(human_faces)  
            human_mesh.compute_vertex_normals()  
            object_vertices = self.get_object_points(i, sampled=True).detach().cpu().numpy()  
            
            transformed_object_mesh = o3d.geometry.TriangleMesh()  
            transformed_object_mesh.vertices = o3d.utility.Vector3dVector(object_vertices)  
            transformed_object_mesh.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=True))  
            transformed_object_mesh.compute_vertex_normals()  
            render.scene.add_geometry("human", human_mesh, human_mat)  
            render.scene.add_geometry("object", transformed_object_mesh, object_mat)  
            render.setup_camera(camera, view_matrix)  
            rendered_img = render.render_to_image()  
            start_frame = (self.start_frame // 2)*2
            frame_path = os.path.join(output_dir, f"frame_{i+start_frame:04d}.png")
            o3d.io.write_image(frame_path, rendered_img)

    def get_object_params(self, frame_idx=None):
        if frame_idx is None:
            frame_idx = self.current_frame
        if self.is_static_object:
            frame_idx = 0

        R_residual = self.get_object_transform(frame_idx)
        R_base = self.base_obj_R[frame_idx]
        
        initial_R_tensor = torch.tensor(self.initial_R[frame_idx], dtype=torch.float32, device=R_residual.device)
        R_final_no_initial = torch.mm(R_residual, R_base)
        R_final = torch.mm(R_final_no_initial, initial_R_tensor)

        t_residual = self.obj_transl_params[frame_idx]
        t_base = self.base_obj_transl[frame_idx]
        
        centers = torch.tensor(self.centers[frame_idx], dtype=torch.float32, device=R_final.device)
        t_final = torch.mv(R_final_no_initial, centers) + torch.mv(R_residual, t_base) + t_residual
        return R_final, t_final
    def get_body_params(self, frame_idx=None):
        """
        获取指定帧的SMPL人体参数
        :param frame_idx: 帧索引，默认为当前帧
        :return: 人体参数
        """
        if frame_idx is None:
            frame_idx = self.current_frame
        self.body_params_sequence["body_pose"][frame_idx] = self.body_pose_params[frame_idx].reshape(1, -1).detach().cpu()
        self.body_params_sequence["betas"][frame_idx] = self.shape_params[frame_idx].reshape(1, -1).detach().cpu()
        self.body_params_sequence["global_orient"][frame_idx] = self.global_orient[frame_idx].reshape(1, 3).detach().cpu()
        self.body_params_sequence["transl"][frame_idx] = self.transl[frame_idx].reshape(1, -1).detach().cpu()
        self.hand_poses[str(frame_idx)]["left_hand"] = self.left_hand_params[frame_idx].reshape(1, -1).detach().cpu().numpy()
        self.hand_poses[str(frame_idx)]["right_hand"] = self.right_hand_params[frame_idx].reshape(1, -1).detach().cpu().numpy()
        return self.body_params_sequence, self.hand_poses
    def get_optimize_result(self):
        R_finals = []
        t_finals = []
        for i in range(self.seq_length):
            body_params, hand_poses = self.get_body_params(i)  
            R_final, t_final = self.get_object_params(i)
            R_finals.append(R_final.detach().cpu().numpy())
            t_finals.append(t_final.detach().cpu().numpy())
            self.body_params_sequence[i] = body_params
            self.hand_poses[i] = hand_poses
        return self.body_params_sequence, self.hand_poses, R_finals, t_finals