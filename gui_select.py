import sys
import open3d as o3d
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton


class PointAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D 点选与红球可视化")
        self.setGeometry(100, 100, 300, 100)

        self.selected_points = []
        self.pcd = self.generate_sample_pcd()  # 用示例点云，可以换成你的mesh表面点

        layout = QVBoxLayout()

        self.open_btn = QPushButton("1. 打开模型并选点")
        self.open_btn.clicked.connect(self.open_and_select_points)
        layout.addWidget(self.open_btn)

        self.show_btn = QPushButton("2. 显示选中点为红球")
        self.show_btn.clicked.connect(self.show_selected_points)
        layout.addWidget(self.show_btn)

        self.setLayout(layout)

    def generate_sample_pcd(self):
        # 生成一个立方体点云用于测试
        mesh = o3d.geometry.TriangleMesh.create_box()
        pcd = mesh.sample_points_poisson_disk(500)
        return pcd

    def open_and_select_points(self):
        print("🔵 打开可交互选择窗口...")
        o3d.visualization.draw_geometries_with_editing([self.pcd])
        # 用户选择点后按'q'退出，点的 index 会保存为 temp文件
        picked = o3d.visualization.read_selection_polygon_volume("selection.json")
        self.selected_points = picked.crop_point_cloud(self.pcd)
        print("✅ 已选择点数量：", len(self.selected_points.points))

    def show_selected_points(self):
        geometries = [self.pcd]

        # 用红球可视化被选中的点
        for point in np.asarray(self.selected_points.points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries, window_name="标注点展示")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PointAnnotator()
    viewer.show()
    sys.exit(app.exec_())
