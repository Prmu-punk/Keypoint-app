from sdf import SDF
import torch
import json
import os
import sys
import numpy as np
import trimesh
import copy
import argparse
import time
from .optimizer_part import VideoBodyObjectOptimizer
import open3d as o3d
import smplx
from .utils.hoi_utils import load_transformation_matrix, update_hand_pose,icp_process
from copy import deepcopy
from probreg import cpd
from tqdm import tqdm
from .hoi_solver import HOISolver
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
def resource_path(relative_path):
    """获取资源文件的绝对路径，PyInstaller兼容"""
    try:
        # PyInstaller创建的临时文件夹路径
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境下的路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# model_type='smpl'
# model_folder="/data/boran/4dhoi/human_motion/TEMOS/SMPL_NEUTRAL.pkl"
# layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}

# model = smplx.create(model_folder, model_type=model_type,
#                          gender='neutral',
#                          num_betas=10,
#                          num_expression_coeffs=10,use_pca=False,use_face_contour=True,flat_hand_mean=True,**layer_arg).cuda()
model_type = 'smplx'
model_folder = resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz")
model = smplx.create(model_folder, model_type=model_type,
                            gender='neutral',
                            num_betas=10,
                            num_expression_coeffs=10,
                            use_pca=False, 
                            flat_hand_mean=True).cuda()
def apply_transform_to_model(vertices, transform_matrix):
    # 顶点转为齐次坐标 
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])

    # 应用变换并返回三维坐标
    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]  

def apply_kalman(param_name, data_list, observation_weight=5.0, transition_cov=0.01):
            """多维度Kalman滤波器"""
            num_frames = len(data_list)
            data_arrays = []
            
            # 转换数据格式
            for data in data_list:
                if hasattr(data, 'numpy'):
                    data_arrays.append(data.numpy())
                else:
                    data_arrays.append(np.array(data))
            
            data_shape = data_arrays[0].shape
            num_dims = data_arrays[0].size
            
            # 将数据重塑为 (num_frames, num_dims) 的矩阵
            observations = np.zeros((num_frames, num_dims), dtype=np.float32)
            for frame_idx in range(num_frames):
                observations[frame_idx] = data_arrays[frame_idx].flatten()
            
            
            # 创建Kalman滤波器
            kf = KalmanFilter(
                initial_state_mean=observations[0],
                initial_state_covariance=np.eye(num_dims) * 0.1,
                transition_matrices=np.eye(num_dims),
                observation_matrices=np.eye(num_dims),
                observation_covariance=np.eye(num_dims) * observation_weight,
                transition_covariance=np.eye(num_dims) * transition_cov,
            )
            smooth_states, _ = kf.smooth(observations)
            filtered_arrays = []
            for frame_idx in range(num_frames):
                filtered_data = smooth_states[frame_idx].reshape(data_shape)
                filtered_arrays.append(filtered_data)
            
            return filtered_arrays
def preprocess_obj(obj_org, object_poses, orient_path, start_frame, end_frame):
    M_trans, Rx, Ry, Rz = load_transformation_matrix(orient_path)
    obj_orgs = []
    for i in range(start_frame, end_frame):
        # 物体pose
        M_t = M_trans[i]
        obj_pcd = deepcopy(obj_org)
        obj_pcd.rotate(Rx, center=obj_pcd.get_center())
        obj_pcd.rotate(Ry, center=obj_pcd.get_center())
        obj_pcd.rotate(Rz, center=obj_pcd.get_center())
        overts = np.asarray(obj_pcd.vertices)
        overts = apply_transform_to_model(overts, M_t)
        obj_pcd.vertices = o3d.utility.Vector3dVector(overts)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        obj_pcd.transform(rot)

        # scale与transl
        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= object_poses['scale']
        new_overts = new_overts - np.mean(new_overts, axis=0)
        new_overts += object_poses['center'][i]
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs


def kp_use(output, hand_poses, obj_orgs, sampled_orgs,
           centers, human_part, K, start_frame, end_frame, joints_to_optimize, video_dir, is_static_object=False):
    # 读物体
    # obj_org = o3d.io.read_triangle_mesh(os.path.join(args.video_dir, 'obj_org.obj'))
    # sampled_obj = obj_org.simplify_quadric_decimation(target_number_of_triangles=1000)
    # 读物体pose
    # with open(os.path.join(args.video_dir, 'output/obj_poses.json')) as f:
    #     object_poses = json.load(f)

    # 关键点配对
    # human_part = json.load(open(f"./data/part_kp.json"))
    body_params = output["smpl_params_incam"]
    global_body_params = output["smpl_params_global"]
    kp_files = sorted(os.listdir(os.path.join(video_dir, 'kp_record')), key=lambda x: int(x.split('.')[0]))
    seq_length = end_frame - start_frame
    object_points_idx = []
    body_points_idx = []
    pairs_2d = []
    object_points = []
    image_points = []

    # 人体参数
    # output = torch.load(args.video_dir + "/motion/result.pt")
    # print(output.keys())
    # body_params = output["smpl_params_incam"]
    # hand_poses = json.load(open(os.path.join(args.video_dir, 'motion/hand_pose.json')))
    # for i in range(args.start_frame, args.end_frame):
    #     body_params["body_pose"][i], hand_poses[str(i)]["left_hand"], hand_poses[str(i)]["right_hand"] \
    #         = update_hand_pose(hand_poses, body_params["global_orient"], body_params["body_pose"], i)

    hoi_solver= HOISolver(model_folder=resource_path('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'))

    for i, file in tqdm(enumerate(kp_files[start_frame:end_frame])):
        annotation = json.load(open(os.path.join(video_dir, 'kp_record', file)))
        if annotation["2D_keypoint"]:
            frame_object_points = []
            frame_image_points = []
            for point_pair in annotation["2D_keypoint"]:
                if is_static_object:
                    frame_object_points.append(np.array(obj_orgs[0].vertices[point_pair[0]] + centers[0]))
                else:
                    frame_object_points.append(np.array(obj_orgs[i].vertices[point_pair[0]] + centers[i]))
                frame_image_points.append(np.array(point_pair[1]))
            object_points.append(np.array(frame_object_points, dtype=np.float32))
            image_points.append(np.array(frame_image_points, dtype=np.float32))

        object_idx = np.zeros((74, 2))
        for k, annot_index in annotation.items():
            if k=="2D_keypoint":
                continue
            human_part_index = list(human_part.keys()).index(k)
            object_idx[human_part_index] = [annot_index, 1]
        pairs_2d.append(annotation["2D_keypoint"])  
        body_idx = [v['index'] for k, v in human_part.items()]
        object_points_idx.append(object_idx)
        body_points_idx.append(body_idx)

        # h_save=o3d.geometry.PointCloud()
        # h_save.points = o3d.utility.Vector3dVector(hpoints)
        # o3d.io.write_point_cloud('./test_h_0.ply', h_save)
        # o3d.io.write_triangle_mesh('test_o_0.obj', obj_init_sample)
        # exit(0)
    hoi_interval = 1
    frames_to_optimize = list(range(0, seq_length, hoi_interval))
    if frames_to_optimize[-1] != seq_length:
        frames_to_optimize.append(seq_length-1) 

    optimized_results = {}
    transform_matrix = []
    joint_mapping=json.load(open(resource_path('video_optimizer/data/joint_reflect.json')))
    
    for i in frames_to_optimize:
        if is_static_object:
            i = 0
        obj_init=copy.deepcopy(obj_orgs[i])
        obj_init_sample=copy.deepcopy(sampled_orgs[i])
        verts, sampled_verts = obj_init.vertices, obj_init_sample.vertices
        verts = verts + centers[i]
        sampled_verts = sampled_verts + centers[i]
        obj_init.vertices = o3d.utility.Vector3dVector(verts)
        obj_init_sample.vertices = o3d.utility.Vector3dVector(sampled_verts)
        
        result=hoi_solver.solve_hoi(obj_init,obj_init_sample,body_params,i,start_frame,end_frame,
                                    hand_poses,object_points_idx,body_points_idx,object_points,image_points,K.cpu().numpy(),joint_mapping)
        body_params['global_orient'][i+ start_frame] = result['global_orient'].detach().cpu()
        body_params['body_pose'][i+ start_frame] = result['body_pose'].detach().cpu()
        transform_matrix.append(result['transform_matrix'])

    if is_static_object:
        first_frame_obj = obj_orgs[0]
        first_frame_sampled = sampled_orgs[0]
        for i in range(seq_length):
            obj_orgs[i] = first_frame_obj
            sampled_orgs[i] = first_frame_sampled
    

    optimizer_args = {"body_params":body_params,
                 "global_body_params":global_body_params,
                 "hand_params":hand_poses,  
                 "object_points_idx":object_points_idx,   
                 "body_points_idx":body_points_idx, 
                 "pairs_2d":pairs_2d, 
                 "object_meshes":obj_orgs, 
                 "sampled_obj_meshes":sampled_orgs, 
                 "centers":centers, 
                 "transform_matrix":transform_matrix,
                 "smpl_model":model,
                 "start_frame":start_frame,
                 "end_frame":end_frame,
                 "video_dir":video_dir,  
                 "lr":0.1,
                 "is_static_object":is_static_object,}
    optimizer = VideoBodyObjectOptimizer(**optimizer_args)
    joints_to_optimize = joints_to_optimize.split(',')
    optimizer.set_mask(joints_to_optimize)
    optimizer.optimize(steps=25, print_every=5, optimize_per_frame=True)
    # optimizer.save_sequence(os.path.join(args.video_dir, 'optimized_sequence'))

    # 渲染视频
    R = torch.eye(3).cuda()
    T = torch.zeros(3).cuda()
    optimizer.create_visualization_video(os.path.join(video_dir, "optimized_frames"), K=K,
                                         R=R, T=T, video_path=os.path.join(video_dir, "optimize_video.mp4"))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Body-Object Optimization")
#     parser.add_argument('--video_dir', type=str, required=True, help='Base directory containing the objects')
#     parser.add_argument('--joints_to_optimize', type=str, required=True,
#                         help='Comma-separated list of joints to optimize')
#     parser.add_argument('--start_frame', type=int, required=True,
#                         help='start frame')
#     parser.add_argument('--end_frame', type=int, required=True,
#                         help='end frame')
#     args = parser.parse_args()
#     main(args)
