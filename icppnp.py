import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def project_points(points, K):
    projected = points @ K.T
    projected /= projected[:, 2:3]
    return projected[:, :2]

def residuals_weighted_priority(x, pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, weight_3d=10.0, weight_2d=1.0):
    rvec = x[:3]
    tvec = x[3:]
    R_mat = R.from_rotvec(rvec).as_matrix()
    pts_3d_3d_trans = (R_mat @ pts_3d_3d_src.T).T + tvec
    res_3d = (pts_3d_3d_trans - pts_3d_3d_tgt).ravel()
    pts_3d_2d_trans = (R_mat @ pts_3d_2d_src.T).T + tvec
    proj_2d = project_points(pts_3d_2d_trans, K)
    res_2d = (proj_2d - pts_2d_tgt).ravel()
    return np.hstack([weight_3d * res_3d, weight_2d * res_2d])

def solve_weighted_priority(pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, weight_3d=10.0, weight_2d=1.0):
    x0 = np.zeros(6)
    res = least_squares(
        residuals_weighted_priority,
        x0,
        args=(pts_3d_3d_src, pts_3d_3d_tgt, pts_3d_2d_src, pts_2d_tgt, K, weight_3d, weight_2d),
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

    R_est, t_est = solve_weighted_priority(src_3d_3d, tgt_3d_3d, src_3d_2d, tgt_2d, K)

    visualize_alignment(src_3d_3d, tgt_3d_3d, R_est, t_est)
    visualize_projection_error(src_3d_2d, tgt_2d, R_est, t_est, K)
