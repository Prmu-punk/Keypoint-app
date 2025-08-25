import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from video_optimizer.utils.camera_utils import transform_to_global

def project_points(points, K, R=np.eye(3), T=np.zeros(3)):
    """
    将世界坐标系下的3D点投影到图像平面

    参数:
        points: np.ndarray, shape (N, 3), 世界坐标系下的3D点
        K: np.ndarray, shape (3, 3), 相机内参
        R: np.ndarray, shape (3, 3), 旋转矩阵
        T: np.ndarray, shape (3,) or (3, 1), 平移向量

    返回:
        np.ndarray, shape (N, 2), 图像坐标 (u, v)
    """
    points = np.asarray(points)
    R = np.asarray(R)
    T = np.asarray(T)
    T = -R.T @ T.reshape(3, 1)

    # 世界坐标 -> 相机坐标
    points_cam = (R @ points.T) + T  # shape (3, N)

    # 相机坐标 ->图像坐标
    homogeneous = (K @ points_cam).T  # shape (N, 3)
    projected = homogeneous / homogeneous[:, 2:3]  # 归一化

    return projected[:, :2]

def residuals_weighted_priority(x, incam_params, global_params, pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, mutiview_info, weight_3d=10.0, weight_2d=1.0):
    rvec = x[:3]
    tvec = x[3:]
    R_mat = R.from_rotvec(rvec).as_matrix()
    residual_all = []

    # 3D-3D error (object to human)
    pts_3d_3d_trans = (R_mat @ pts_3d_3d_src.T).T + tvec
    res_3d = (pts_3d_3d_trans - pts_3d_3d_tgt).ravel()
    residual_all.append(weight_3d
     * res_3d)
    # 3D-2D error (object to image)
    # 1. Transform object points by the optimized pose
    if K is not None:
        pts_3d_2d_world = (R_mat @ pts_3d_2d_src.T).T + tvec    
        # 3. Project to 2D
        proj_2d = project_points(pts_3d_2d_world, K)
        res_2d = (proj_2d - pts_2d_tgt).ravel()
        residual_all.append(weight_2d * res_2d)
    if mutiview_info is not None:
        i = 0
        for src_3d_2d, tgt_2d, cam_params in zip(*mutiview_info):
            # save_points_as_obj(src_3d_2d, f"begin_transform_{i}.obj")
            wld_3d_2d = (R_mat @ src_3d_2d.T).T + tvec
            
            # 保存transform_to_global变换前的点
            # save_points_as_obj(wld_3d_2d, f"before_transform_{i}.obj")
            
            _, wld_3d_2d = transform_to_global(incam_params, global_params, overts=wld_3d_2d)
            
            # 保存transform_to_global变换后的点
            # save_points_as_obj(wld_3d_2d, f"after_transform_{i}.obj")
            
            cam_R = np.array(cam_params['R'])
            cam_t = np.array(cam_params['T'])
            cam_K = np.array(cam_params['K'])
            # save_points_as_obj(cam_t[np.newaxis, :], f"cam_t_{i}.obj")
            proj_2d = project_points(wld_3d_2d, cam_K, cam_R, cam_t)
            
            # 可视化proj_2d在600*800画布上
            # width_px = 800
            # height_px = 600
            # dpi = 100
            # figsize = (width_px/dpi, height_px/dpi)
            
            # plt.figure(figsize=figsize, dpi=dpi)
            # plt.xlim(0, 800)
            # plt.ylim(0, 600)
            # plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c='red', s=50, label='投影点', marker='o')
            # plt.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c='blue', s=50, label='目标点', marker='x')
            # plt.legend()
            # plt.title(f'视图 {i} - 投影结果')
            # plt.xlabel('X (pixels)')
            # plt.ylabel('Y (pixels)')
            # plt.grid(True, alpha=0.3)
            # plt.gca().invert_yaxis()  # 图像坐标系y轴向下
            # plt.savefig(f'view_{i}_projection.png', dpi=dpi, bbox_inches='tight')
            # plt.close()
            
            residual_all.append(weight_2d * (proj_2d - tgt_2d).ravel())
            # i += 1
    return np.hstack(residual_all)

def solve_weighted_priority(incam_params, global_params, pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, multiview_info, weight_3d=10.0, weight_2d=1.0):
    print("Starting optimization...",pts_3d_3d_src.shape, pts_3d_3d_tgt.shape, pts_3d_2d_src.shape, pts_2d_tgt.shape,multiview_info)
    # x0 = np.zeros(6)
    x0 = np.zeros(6)
    residuals = residuals_weighted_priority(
        x0,
        incam_params, global_params,
        pts_3d_3d_src, pts_3d_3d_tgt,
        pts_3d_2d_src, pts_2d_tgt,
        K, multiview_info, weight_3d, weight_2d
    )
    print(f"[DEBUG] Number of variables: {len(x0)}")
    print(f"[DEBUG] Number of residuals: {len(residuals)}")

    res = least_squares(
        residuals_weighted_priority,
        x0,
        args=(incam_params, global_params, pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, multiview_info, weight_3d, weight_2d),
        method='lm'
    )
    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:]
    R_opt = R.from_rotvec(rvec_opt).as_matrix()
    return R_opt, tvec_opt

def visualize_projection_error(pts_3d_2d_src, pts_2d_tgt, R_opt, t_opt, K, image=None):
    def project_points(pts_3d, K):
        proj = pts_3d @ K.T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    pts_3d_cam = (R_opt @ pts_3d_2d_src.T).T + t_opt
    pts_2d_proj = project_points(pts_3d_cam, K)

    plt.figure(figsize=(8, 6))
    if image is not None:
        plt.imshow(image)
    else:
        plt.gca().invert_yaxis()  # 符合图像坐标习惯

    # Ground truth 2D (红色)
    plt.scatter(pts_2d_tgt[:, 0], pts_2d_tgt[:, 1], c='r', label='Target 2D', s=50)
    # Projected 2D (蓝色)
    plt.scatter(pts_2d_proj[:, 0], pts_2d_proj[:, 1], c='b', label='Projected 2D', s=50)

    # 误差线
    for i in range(len(pts_2d_tgt)):
        plt.plot(
            [pts_2d_tgt[i, 0], pts_2d_proj[i, 0]],
            [pts_2d_tgt[i, 1], pts_2d_proj[i, 1]],
            'gray', linestyle='--', linewidth=1
        )

    plt.legend()
    plt.title("2D Projection Error")
    plt.xlabel("u (pixels)")
    plt.ylabel("v (pixels")
    plt.grid(True)
    plt.show()
def visualize_alignment(pts_src, pts_tgt, R_opt, t_opt):
    pts_src_trans = (R_opt @ pts_src.T).T + t_opt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_src[:, 0], pts_src[:, 1], pts_src[:, 2], c='g', label='Src 3D (before)', s=50)
    ax.scatter(pts_src_trans[:, 0], pts_src_trans[:, 1], pts_src_trans[:, 2], c='b', label='Src 3D (after)', s=50)
    ax.scatter(pts_tgt[:, 0], pts_tgt[:, 1], pts_tgt[:, 2], c='r', label='Target 3D', s=50)
    for i in range(len(pts_src)):
        ax.plot(
            [pts_src_trans[i, 0], pts_tgt[i, 0]],
            [pts_src_trans[i, 1], pts_tgt[i, 1]],
            [pts_src_trans[i, 2], pts_tgt[i, 2]],
            c='gray', linestyle='--', linewidth=1
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("3D Alignment Visualization")
    plt.tight_layout()
    plt.show()

def save_points_as_obj(points, filename):
    """简单保存3D点为.obj格式"""
    with open(filename, 'w') as f:
        for i, point in enumerate(points):
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    print(f"已保存: {filename}")

# 测试样例
if __name__ == "__main__":
    np.random.seed(42)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    n, m = 4, 3  # 更少点

    src_3d_3d = np.random.randn(n, 3) * 0.3 + [0, 0, 4]
    src_3d_2d = np.random.randn(m, 3) * 0.3 + [0, 0, 4]

    R_gt = R.from_euler("zyx", [5, -10, 15], degrees=True).as_matrix()
    t_gt = np.array([0.2, -0.1, 0.3])

    tgt_3d_3d = (R_gt @ src_3d_3d.T).T + t_gt + np.random.randn(n, 3) * 0.01
    tgt_2d = project_points((R_gt @ src_3d_2d.T).T + t_gt, K)
    tgt_2d += np.random.randn(m, 2) * 0.5

    R_est, t_est = solve_weighted_priority(src_3d_3d, tgt_3d_3d, src_3d_2d, tgt_2d, K, np.eye(3), np.zeros(3))

    visualize_alignment(src_3d_3d, tgt_3d_3d, R_est, t_est)
    visualize_projection_error(src_3d_2d, tgt_2d, R_est, t_est, K)
