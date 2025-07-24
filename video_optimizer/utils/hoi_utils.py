import json

import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import scipy
from scipy.spatial.transform import Rotation as R
import smplx
import open3d as o3d
from probreg import cpd

# import torchvision.transforms as transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def reconstruct3D_from_depth(pred_depth):
    # cam_u0 = rgb.shape[1] / 2.0
    # cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5
    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax
    depth_scale_1 = pred_depth_norm
    return depth_scale_1


def reconstruct_3D(masks, depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        # print('Infinit focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
        # print(depth.max())
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth
    z[masks == 0] = -1
    pcd_new = []
    x = np.reshape(x, (width * height, 1)).astype(np.float32)
    y = np.reshape(y, (width * height, 1)).astype(np.float32)
    z = np.reshape(z, (width * height, 1)).astype(np.float32)
    # pcd = np.concatenate((x, y, z), axis=1)
    pcd = np.concatenate((x, y, z), axis=1)
    # print(pcd.shape)
    index = np.asarray(pcd[:, 2] >= 0)
    # Filter out rows where the z value is negative
    pcd_new = pcd[pcd[:, 2] >= 0]

    pcd_new[:,2]*=-1

    # index=index.reshape(width,height)

    # pcd_new already has the shape (-1, 3), no need to reshape

    return pcd_new, index


def reconstruct_3D_lucid(masks, depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    W = depth.shape[1]
    H = depth.shape[0]
    focal = (1.8269e+02, 1.8269e+02)
    fov = (2 * np.arctan(W / (2 * focal[0])), 2 * np.arctan(H / (2 * focal[1])))
    K = np.array([
        [focal[0], 0., W / 2],
        [0., focal[1], H / 2],
        [0., 0., 1.],
    ]).astype(np.float32)
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pts_coord_cam = np.matmul(np.linalg.inv(K),
                              np.stack((x * depth, y * depth, 1 * depth), axis=0).reshape(3, -1)).transpose(1, 0)
    # new_pts_colors2 = (np.array(image_curr).reshape(-1, 3).astype(np.float32) / 255.)  ## new_pts_colors2
    # z[masks==0]=-1
    # pcd_new=[]
    # x = np.reshape(x, (width * height, 1)).astype(np.float32)
    # y = np.reshape(y, (width * height, 1)).astype(np.float32)
    # z = np.reshape(z, (width * height, 1)).astype(np.float32)
    # # pcd = np.concatenate((x, y, z), axis=1)
    # pcd = np.concatenate((x, y, z), axis=1)
    # # print(pcd.shape)
    # index = np.asarray(pcd[:, 2] >= 0)
    # # Filter out rows where the z value is negative
    # pcd_new = pcd[pcd[:, 2] >= 0]

    # index=index.reshape(width,height)

    # pcd_new already has the shape (-1, 3), no need to reshape
    index = masks.reshape(-1)
    pcd_new = pts_coord_cam[index == 1]

    return pcd_new, index == 1


def reconstruct_depth(masks, depth, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    depth = np.squeeze(depth)
    # print(depth.shape)
    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd, index = reconstruct_3D(masks, depth, f=focal)
    # pcd,index = reconstruct_3D_lucid(masks,depth, f=focal)
    # save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))
    # print(pcd.shape)
    return pcd, index


def get_obj_pcd(masks, depth):
    pred_depth = depth
    # pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

    # recover focal length, shift, and scale-invariant depth
    depth_scaleinv = reconstruct3D_from_depth(pred_depth)
    # mask_obj_all=np.zeros((masks.shape[-2],masks.shape[-1]), dtype=bool)
    # # for mask in masks:
    # # for obj in masks:
    # mask_obj_all[]=True
    pcd, index = reconstruct_depth(masks, depth_scaleinv, focal=1e6)
    # findex=get_front_index(pcd)
    # index[findex]=False
    return pcd, index

def project(xyz, K):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    # xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy
    
    
def get_scene_pcd(depth):
    pred_depth = depth
    # pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
    # recover focal length, shift, and scale-invariant depth
    depth_scaleinv = reconstruct3D_from_depth(pred_depth)

    mask_obj_all = np.ones((depth.shape[0], depth.shape[1]), dtype=bool)
    pcd, index = reconstruct_depth(mask_obj_all, depth_scaleinv, focal=1e6)
    return pcd, index


def point_align_vis(body, h, w, K):


    vert = np.array(body)

    # homog_coord = np.ones(list(vert.shape[:-1]) + [1])
    # points_h = np.concatenate([vert, homog_coord], axis=-1)
    # projected = points_h

    img_point=project(vert,np.asarray(K))

    img_point = img_point.astype(np.int32)

    # find verts projected on the image
    flag1 = np.logical_and(img_point[:, 0] < w, img_point[:, 0] >= 0)
    flag2 = np.logical_and(img_point[:, 1] < h, img_point[:, 1] >= 0)
    filter = np.where(np.logical_and(flag1, flag2))[0]
    rest = vert[filter, :] #vert是human的点
    img_point = img_point[filter, :]
    index = np.argsort(rest[:, 2])  # sort by depth
    align = img_point[:, 1] * w + img_point[:, 0]  # pixel corresponding
    u_all = align
    u, indices = np.unique(align[index], return_index=True)  # find the first appearance（按照深度，实际上不做这一步也没有影响，公式可以不体现）
    # print('indices', indices.shape)
    front = rest[index[indices], :]
    pid = filter[index[indices]]
    l = len(front[:, 2])
    final = np.argsort(front[:, 2]) #取了前一半的点，认为这些点是人的前半身
    return u[final], front[final], pid[
        final]  # pixel index, corresponding point location obtained from human, corresponding pixel index


def get_front_index(points):
    fz = np.percentile(points[:, 2], 80)
    mean_z = np.mean(fz)
    # print(mean_z)
    # mean_z=2300
    return np.where(points[:, 2] < mean_z)[0]


def align(scene, body, h, w, K):


    # avgz = np.mean(body[:, 2])
    #
    # body[:, :2] += np.array([[avgz * center[0] / focal_len, avgz * center[1] / focal_len]])
    num_vert = body.shape[0]

    pidx, front_h, vidx = point_align_vis(body, h, w, K)


    front_s = scene[pidx, :]
    # f_index = get_front_index(front_s)
    # print(len(f_index))
    # front_s= front_s[f_index, :]

    # front_s = scene_vertices
    # find scale
    # cur_sel = np.where(np.logical_and(vidx >= 0, vidx < 1 * num_vert))[0]
    b = front_h
    s = front_s
    dis_b = np.mean(scipy.spatial.distance.cdist(b, b))
    dis_s = np.mean(scipy.spatial.distance.cdist(s, s))
    scale = dis_s / dis_b
    b *= scale
    displace = np.mean(s, axis=0) - np.mean(b, axis=0)
    b += displace
    return scale, displace, front_s, b, pidx


def load_transformation_matrix(t_dir):
    T=json.load(open(t_dir+'transform.json'))
    T = np.array(T)
    rotate=json.load(open(t_dir+'rotate90.json'))
    Rx, Ry, Rz = rotate

    return T, Rx, Ry, Rz

def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation

M = np.diag([-1, 1, 1])
def update_hand_pose(hand_poses,global_orient,body_params,frame_idx):

    body_pose = body_params[frame_idx].detach().cpu().numpy().reshape(1, -1)
    global_orient = global_orient[frame_idx].detach().cpu().numpy().reshape(1, 3)
    try:
        handpose=hand_poses[str(frame_idx)]
    except:
        return torch.from_numpy(body_pose), np.zeros(45), np.zeros(45)
    full_body_pose = np.concatenate(
        [global_orient.reshape(1, 3), body_pose.reshape(21, 3)], axis=0)
    left_elbow_global_rot = compute_global_rotation(full_body_pose, 18)  # left elbow IDX: 18
    right_elbow_global_rot = compute_global_rotation(full_body_pose, 19)  # left elbow IDX: 19

    if 'left_hand' in handpose:
        global_orient_hand_left = np.asarray(handpose["left_global_orient"]).reshape(3, 3)
        left_wrist_global_rot = M @ global_orient_hand_left @ M  # mirror switch
        left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
        left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()
        body_pose[:, 57:60] = left_wrist_pose_vec
    # global_orient_hand_left=np.asarray(hand_poses[str(frame_idx)]["global_orient"][0]).reshape(3,3)
    # print(global_orient_hand_left.shape)
    # exit(0)
    if 'right_hand' in handpose:
        global_orient_hand_right = np.asarray(handpose["right_global_orient"]).reshape(3, 3)  
        right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ global_orient_hand_right
        right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()
        body_pose[:, 60:63] = right_wrist_pose_vec


    left_hand_pose = np.zeros(45)
    right_hand_pose = np.zeros(45)
    for i in range(15):
        if 'left_hand' in handpose:
            left_finger_pose = M @ np.asarray(hand_poses[str(frame_idx)]["left_hand"])[
                i] @ M
            left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
            left_hand_pose[i * 3: i * 3 + 3] = left_finger_pose_vec
        if 'right_hand' in handpose:
            right_finger_pose = np.asarray(hand_poses[str(frame_idx)]["right_hand"][i])
            right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
            right_hand_pose[i * 3: i * 3 + 3] = right_finger_pose_vec

    return torch.from_numpy(body_pose), left_hand_pose, right_hand_pose

def apply_transform_to_model(vertices, transform_matrix):
    # 顶点转为齐次坐标
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])

    # 应用变换并返回三维坐标
    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]  # 透视除法


def preprocess_obj(obj_org, object_poses, orient_path, seq_length):
    M_trans, Rx, Ry, Rz = load_transformation_matrix(orient_path)
    obj_orgs = []
    for i in range(seq_length):
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


def get_corresponding_point(object_points_idx, body_points_idx, body_points, object_mesh):
    """
    获取指定帧的对应点
    :param frame_idx: 帧索引，默认为当前帧
    :return: 人体点和物体点的对应关系字典
    """

    # 获取在交互的人体点索引

    interacting_indices = object_points_idx[:, 1] != 0
    interacting_body_indices = np.asarray(body_points_idx)[interacting_indices]

    # 获取对应的人体点坐标
    # time_body = time.time()
    body_points = body_points[interacting_body_indices]
    # time_body2 = time.time()
    # print("get_body_points:", time_body2-time_body)

    # 获取对应的物体点坐标
    object_points = torch.tensor(np.array(object_mesh.vertices),
                                 device=body_points.device).float()
    object_points = object_points
    obj_index = object_points_idx[interacting_indices][:, 0]
    interactiong_obj = object_points[obj_index]

    # 创建对应点字典
    corresponding_points = {
        'body_points': body_points,
        'object_points': interactiong_obj
    }

    return corresponding_points
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def rigid_transform_svd_with_corr(A, B):
    A_corr = A
    B_corr = B
    centroid_A = A_corr.mean(axis=0)
    centroid_B = B_corr.mean(axis=0)
    AA = A_corr - centroid_A
    BB = B_corr - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T
    t = centroid_B - R_mat @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def residuals_with_corr(x, A, B):
    rot_vec = x[:3]
    t = x[3:]
    R_mat = R.from_rotvec(rot_vec).as_matrix()
    A_sub = A
    B_sub = B
    A_trans = (R_mat @ A_sub.T).T + t
    return (A_trans - B_sub).ravel()

def refine_rigid_with_corr(A, B, x0=None):
    if x0 is None:
        T0 = rigid_transform_svd_with_corr(A, B)
        rot0 = R.from_matrix(T0[:3, :3]).as_rotvec()
        t0 = T0[:3, 3]
        x0 = np.hstack([rot0, t0])
    res = least_squares(residuals_with_corr, x0, args=(A, B))
    R_opt = R.from_rotvec(res.x[:3]).as_matrix()
    t_opt = res.x[3:]
    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt
    return T_opt

def icp_process(object_points_idx, body_points_idx,hpoints,obj_init,obj_init_sample):
    corresp = get_corresponding_point(object_points_idx, body_points_idx, hpoints, obj_init)
    org_overts = np.asarray(obj_init.vertices)
    org_sample_verts=np.asarray(obj_init_sample.vertices)
    # print(hpoints.shape, org_overts.shape, org_sample_verts.shape)
    # org_o = o3d.geometry.PointCloud()
    # org_o.points = o3d.utility.Vector3dVector(org_overts)
    # overts_sample = o3d.geometry.PointCloud()
    # overts_sample.points = o3d.utility.Vector3dVector(org_sample_verts)

    # hp = o3d.geometry.PointCloud()
    # hp.points = o3d.utility.Vector3dVector(hpoints)

    # hverts = o3d.geometry.PointCloud()
    # hverts.points = o3d.utility.Vector3dVector(corresp['body_points'])
    # overts = o3d.geometry.PointCloud()
    # overts.points = o3d.utility.Vector3dVector(corresp['object_points'])

    source_points = np.asarray(corresp['object_points'])
    target_points = np.asarray(corresp['body_points'])
    print(source_points.shape, target_points.shape)

    # tf_param, _, _ = cpd.registration_cpd(overts, hverts, maxiter=100, tol=0.01)
    T_est = refine_rigid_with_corr(source_points,target_points)

    org_overts_h = np.hstack([org_overts, np.ones((org_overts.shape[0], 1))])
    transformed_org_o = (T_est @ org_overts_h.T).T[:, :3]

    org_overts_h_sample= np.hstack([org_sample_verts, np.ones((org_sample_verts.shape[0], 1))])
    transformed_overts_sample = (T_est @ org_overts_h_sample.T).T[:, :3]


    # overts.points = tf_param.transform(overts.points)
    # org_o.points = tf_param.transform(org_o.points)
    # overts_sample.points=tf_param.transform(overts_sample.points)

    return transformed_org_o,transformed_overts_sample






