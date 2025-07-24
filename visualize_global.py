import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import numpy as np
import argparse
import smplx
from einops import einsum, rearrange
import torch.nn.functional as F
from global_utils.utils import Renderer, get_global_cameras_static, get_ground_params_from_points
def resource_path(relative_path):
    try:
        # PyInstaller创建的临时文件夹路径
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境下的路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
def apply_T_on_points(points, T):
    """
    Args:
        points: (..., N, 3)
        T: (..., 4, 4)
    Returns: (..., N, 3)
    """
    points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T
def transform_mat(R, t):
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

def compute_T_ayfz2ay(joints, inverse=False):
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
        return transform_mat(R_ay2ayfz, t_ay2ayfz)
    else:
        return transform_mat(R_ayfz2ay, t_ayfz2ay)

def get_global(verts):  
    J_regressor = torch.load(resource_path("J_regressor.pt")).double().cuda()  
    
    def move_to_start_point_face_z(verts):  
        "XZ to origin, Start from the ground, Face-Z"  
        verts = verts.clone().cuda()  # (L, V, 3)  
        
        # 提取人体部分顶点(假设前10475个顶点为SMPL-X人体顶点)  
        human_vertex_count = 10475  
        human_part = verts[:, :human_vertex_count, :]  
        offset = einsum(J_regressor, human_part[0], "j v, v i -> j i")[0]  
        offset[1] = human_part[:, :, [1]].min()  
        verts = verts - offset  
        T_ay2ayfz = compute_T_ayfz2ay(  
            einsum(J_regressor, human_part[[0]], "j v, l v i -> l j i"),   
            inverse=True  
        )  
        verts = apply_T_on_points(verts, T_ay2ayfz)  
        return verts  
    
    verts_glob = move_to_start_point_face_z(verts)  
    
    # 只使用人体部分顶点计算关节  
    human_vertex_count = 10475  
    human_part = verts_glob[:, :human_vertex_count, :]  
    joints_glob = einsum(J_regressor, human_part, "j v, l v i -> l j i")  # (L, J, 3)  
    
    global_R, global_T, global_lights = get_global_cameras_static(  
        human_part.float().cpu(),  
        beta=2.0,  
        cam_height_degree=20,  
        target_center_height=1.0,  
    )  
    return global_R, global_T, global_lights

    # -- rendering code -- #  
    # length, width, height = get_video_lwh(video_path)  
    # _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens   
    # renderer = Renderer(width, height, device="cuda", faces=faces_smplx, K=K)  
    # scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], human_part)  
    # renderer.set_ground(scale * 1.5, cx, cz)  
    # color = torch.ones(3).float().cuda() * 0.8  

    # debug_cam = False
    # render_length = length if not debug_cam else 8  
    # writer = get_writer(output_path, fps=30, crf=23)  
    # for i in tqdm(range(0,render_length-int(args.start_idx)), desc=f"Rendering Global"):  
    #     cameras = renderer.create_camera(global_R[i], global_T[i])  
    #     # 使用整个顶点集(包括人体和物体)进行渲染  
    #     img = renderer.render_with_ground(verts_glob[[i]].float(), color[None], cameras, global_lights)  
    #     writer.write_frame(img)  
    # writer.close()  


# output=torch.load("/data/boran/4dhoi/Dataset/Ways_to_Jump_+_Sit_+_Fall_Cold_clip2/motion/result.pt")
# global_param=output['smpl_params_global']
#
# # initialize SMPL-X model
# model_type='smplx'
# model_folder="/data/boran/4dhoi/human_motion/TEMOS/SMPLX_NEUTRAL.npz"
# layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
# hand_poses=json.load(open("/data/boran/4dhoi/Dataset/Ways_to_Jump_+_Sit_+_Fall_Cold_clip2/motion/hand_pose.json"))
# model = smplx.create(model_folder, model_type=model_type,
#                          gender='neutral',
#                          num_betas=10,
#                          num_expression_coeffs=10,use_pca=False,use_face_contour=True,flat_hand_mean=True,**layer_arg).cuda()
# faces = model.faces
#
#
# for k,v in global_param.items():
#     print(k,v.shape)
# body_pose=global_param['body_pose']
# global_orient=global_param['global_orient']
# betas=global_param['betas']
# transl=global_param['transl']
#
# zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
#
# t_len=body_pose.shape[0]
# verts_list=[]
# for t in range(t_len):
#
#     handpose=hand_poses[str(t)]
#     if 'left_hand' in handpose:
#         left_hand_pose = np.asarray(handpose['left_hand'])
#         left_hand_pose = matrix_to_axis_angle(left_hand_pose)
#         left_hand_pose[:, 1:] *= -1
#         left_hand_pose=torch.from_numpy(left_hand_pose).float().cuda()
#     else:
#         left_hand_pose = torch.zeros((1, 45)).float().cuda()
#         print("Use smplx left hand pose")
#     if 'right_hand' in handpose:
#         right_hand_pose = np.asarray(handpose['right_hand'])
#         right_hand_pose = matrix_to_axis_angle(right_hand_pose)
#         right_hand_pose=torch.from_numpy(right_hand_pose).float().cuda()
#     else:
#         right_hand_pose =  torch.zeros((1, 45)).float().cuda()
#         print("Use smplx right hand pose")
#     params = {"global_orient": global_orient[t].reshape(1, -1).cuda(),
#               "body_pose": body_pose[t].reshape(1, -1).cuda(),
#               "betas": betas[t].reshape(1, -1).cuda(),
#               "expression": torch.zeros((1, 10)).float().cuda(),
#               "left_hand_pose":left_hand_pose,
#               "right_hand_pose":right_hand_pose,
#               "jaw_pose": zero_pose,
#               "leye_pose": zero_pose,
#               "reye_pose": zero_pose,
#               "transl": transl[t].reshape(1, -1).cuda()}
#
#     output = model(**params)
#     verts_list.append(output.vertices[0])
# verts_list = torch.stack(verts_list)
# print(verts_list.shape)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process 3D objects.')
#     parser.add_argument('--video_dir', type=str, help='Base directory containing the objects',
#                         default="/data/boran/4dhoi/Dataset/Ways_to_Jump_+_Sit_+_Fall_Cold_clip2/")
#     parser.add_argument('--start_idx', type=int, default=0, help='Start index for rendering')
#     parser.add_argument('--video_names', type=str, default='', help='Names of the videos to process')
#     args = parser.parse_args()

#     video_names = args.video_names.split(',') if args.video_names else []
#     for video_name in video_names:
#         video_dir = os.path.join(args.video_dir, video_name)

#         verts_list = torch.load(os.path.join(video_dir,"output/all_frames.pt"))
#         # print(verts_list.shape)
#         combined_faces = np.load(os.path.join(video_dir,"output/combined_faces.npy"))
#         # print(combined_faces.shape)
#         output_path= os.path.join(video_dir, "output", "global.mp4")
#         render_global(verts_list, combined_faces, output_path,
#                     os.path.join(video_dir, "video.mp4"))
