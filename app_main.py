
import tkinter as tk
import argparse
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import numpy as np
import torch
torch.cuda.init()
import open3d as o3d
import os
import sys
import smplx
import trimesh
import shutil
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
import platform
import threading
import time

from video_optimizer.utils.hoi_utils import load_transformation_matrix, update_hand_pose
from video_optimizer.utils.parameter_transform import transform_and_save_parameters
from video_optimizer.kp_use import kp_use, kp_use_multiview
from video_optimizer.utils.camera_utils import create_camera_for_object, transform_to_global, rotate_camera

from copy import deepcopy
from rotate_smpl import matrix_to_axis_angle
from icppnp import solve_weighted_priority
from pykalman import KalmanFilter

# CoTracker imports
try:
    from cotracker.predictor import CoTrackerOnlinePredictor
    from cotracker.utils.visualizer import Visualizer
    import imageio.v3 as iio
    COTRACKER_AVAILABLE = True
    print("CoTracker Online is available")
except ImportError as e:
    COTRACKER_AVAILABLE = False
    print(f"Warning: CoTracker not available: {e}")
    print("Will fall back to interpolation for 2D keypoints")


EMOJI_FONT = "mincho"
DEFAULT_FONT = (EMOJI_FONT, 10)
TITLE_FONT = (EMOJI_FONT, 16, "bold")
SUBTITLE_FONT = (EMOJI_FONT, 12, "bold")
BUTTON_FONT = (EMOJI_FONT, 9)
INFO_FONT = (EMOJI_FONT, 10, "bold")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model_type = 'smplx'
model_folder = resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz")
model = smplx.create(model_folder, model_type=model_type,
                            gender='neutral',
                            num_betas=10,
                            num_expression_coeffs=10,
                            use_pca=False, 
                            flat_hand_mean=True)

def load_transformation_matrix(t_dir):
    T=json.load(open(t_dir+'transform.json'))
    T = np.array(T)
    rotate=json.load(open(t_dir+'rotate90.json'))
    Rx, Ry, Rz = rotate
    return T, Rx, Ry, Rz

def apply_transform_to_model(vertices, transform_matrix):
    # 顶点转为齐次坐标
    homogenous_verts = np.hstack([vertices, np.ones((len(vertices), 1))])

    # 应用变换并返回三维坐标
    transformed = (transform_matrix @ homogenous_verts.T).T
    return transformed[:, :3] / transformed[:, [3]]  # 透视除法

def preprocess_obj(obj_org, object_poses, orient_path, seq_length):
    centers = np.array(object_poses['center'])
    obj_orgs = []
    for i in range(seq_length):
        obj_pcd=deepcopy(obj_org)
        if 'rotation' in object_poses:
            rotation_matrix = object_poses['rotation'][i]
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            obj_pcd.transform(transform_matrix)
        new_overts = np.asarray(obj_pcd.vertices)
        new_overts *= object_poses['scale']
        new_overts = new_overts - np.mean(new_overts, axis=0)
        # new_overts += centers[i]
        obj_pcd.vertices = o3d.utility.Vector3dVector(new_overts)
        obj_orgs.append(obj_pcd)
    return obj_orgs, centers

class KeyPointApp:
    def __init__(self, root, args):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.title("KeyPoint Annotation Tool")

        self.style = ttk_boot.Style(theme="superhero")
        self.configure_emoji_fonts()

        self.obj_point = None
        self.kp_pair={"2D_keypoint":[]}
        self.human_joint_positions = {}
        self.annotated_frames = set()
        self.annotated_frames_2D = set()
        self.render_key_frame = set()
        self.last_frame = None
        self.last_frame_2D = None
        self.no_annot = True
        self.no_annote_2D = True
        self.selected_human_kp = None
        self.selected_2d_point = None
        self.is_static_object = tk.BooleanVar(value=False)
        
        # CoTracker相关变量

        self.tracked_points = {}  # 存储追踪结果 {obj_idx: [(x, y), ...]}
        self.tracking_active = False
        self.video_frames = None  # 存储视频帧数据
        
        # 缓存所有帧的关键点数据，避免重复读取文件
        self.frame_keypoints_cache = {}  # {frame_idx: {"2D_keypoint": [...], ...}}
        
        # 暂存优化参数，用于最终整合保存
        self.temp_optimized_params = []  # 存储每次优化的参数
        

        self.cotracker_model = CoTrackerOnlinePredictor(checkpoint='./co-tracker/checkpoints/scaled_online.pth').cuda()
        self.video_tensor = None
        self.tracked_points = {}  # 存储追踪结果 {obj_idx: [(x, y), ...]}
        self.tracking_active = False

        self.rendered_frames = set()
        self.modify_obj_point = None
        self.selected_multiview_2d_point = None
        self.annotated_modified_frames = set()
        self.last_modified_frame = None
        self.current_multiview_cam_params = None
        

        self.load_config_files()
        self.setup_ui()
        self.load_data(args)

        # self.show_video_interval(0, 10)
    
    def configure_emoji_fonts(self):
        """配置支持emoji的字体"""

        # 配置ttk样式的字体
        self.style.configure("TLabel", font=DEFAULT_FONT)
        self.style.configure("TButton", font=BUTTON_FONT)
        self.style.configure("TCheckbutton", font=DEFAULT_FONT)
        self.style.configure("TLabelframe.Label", font=SUBTITLE_FONT)

        self.style.configure("Title.TLabel", font=TITLE_FONT)
        self.style.configure("Info.TLabel", font=INFO_FONT)
            
        
    def load_config_files(self):
        """加载配置文件"""
        with open(resource_path("part_kp.json"), "r") as file:
            self.all_joint = json.load(file)
        for joint, value in self.all_joint.items():
            self.human_joint_positions[joint] = value['point']

        with open(resource_path("main_joint.json"), "r") as file:
            self.main_joint_coord = json.load(file)
        with open(resource_path("joint_tree.json"), "r") as file:
            self.joint_tree = json.load(file)
        with open(resource_path("button_name.json"), "r", encoding='utf-8') as file:
            self.button_name = json.load(file)
    
    def setup_ui(self):
        """设置用户界面"""
        self.main_container = ttk_boot.Frame(self.root, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.left_container = ttk_boot.Frame(self.main_container)
        self.left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.top_frame = ttk_boot.Frame(self.left_container)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_frame = ttk_boot.LabelFrame(self.top_frame, text="Target video", padding=15, bootstyle=PRIMARY)
        self.plot_frame = ttk_boot.LabelFrame(self.top_frame, text="3D model", padding=15, bootstyle=SUCCESS)
        self.human_image_frame = ttk_boot.LabelFrame(self.top_frame, text="Human keypoints", padding=15, bootstyle=INFO)
        
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 5))
        self.human_image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 0))

        self.bottom_frame = ttk_boot.LabelFrame(self.left_container, text="Reference view", padding=10, bootstyle=SECONDARY)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(10, 0))

        self.render_frame = ttk_boot.LabelFrame(self.main_container, text="Render video", padding=15, bootstyle=WARNING)
        self.render_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.setup_video_panel()
        self.setup_3d_control_panel()
        self.setup_human_panel()
        self.setup_render_panel()

        
        
    def setup_video_panel(self):
        """设置视频面板"""

        self.video_display_frame = ttk_boot.Frame(self.video_frame)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_frame = ttk_boot.Frame(self.video_frame)
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk_boot.Scale(
            self.progress_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self.seek_video,
            bootstyle=PRIMARY
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.frame_label = ttk_boot.Label(
            self.progress_frame, 
            text="0/0", 
            font=INFO_FONT,
            bootstyle=PRIMARY
        )
        self.frame_label.pack(side=tk.RIGHT)
        
        # 按钮
        self.controls_frame = ttk_boot.Frame(self.video_frame)
        self.controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.play_button = ttk_boot.Button(
            self.controls_frame, 
            text="Play", 
            command=self.toggle_video,
            bootstyle="success",
            width=15
        )
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        self.static_checkbox = ttk_boot.Checkbutton(
            self.controls_frame,
            text="Static object",
            variable=self.is_static_object,
            bootstyle="success-round-toggle"
        )
        self.static_checkbox.pack(side=tk.RIGHT)
        
    def setup_3d_control_panel(self):
        """设置3D控制面板"""
        self.button_frame = ttk_boot.Frame(self.plot_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 10))
         # 这个按钮只在渲染过的帧上可用

        
        reset_btn = ttk_boot.Button(
            self.button_frame,
            text="Reset keypoints",
            command=self.reset_keypoints,
            bootstyle="outline-danger",
            width=20
        )
        reset_btn.pack(fill=tk.X, pady=2)

        # Select 3D point 按钮应该总是可用的
        select_3d_btn = ttk_boot.Button(
            self.button_frame,
            text="Select 3D point",
            command=self.open_o3d_viewer,
            bootstyle="primary",
            width=20
        )
        select_3d_btn.pack(fill=tk.X, pady=2)
        self.annotation_buttons = []
        buttons = [
            ("Select 2D point", self.keypoint_2D, "outline-info"),
            ("Start tracking", self.start_2d_tracking, "success"),
            ("Re-track from current", self.restart_tracking_from_current, "outline-success")
        ]
        
        for i, (text, command, style) in enumerate(buttons):
            btn = ttk_boot.Button(
                self.button_frame,
                text=text,
                command=command,
                bootstyle=style,
                width=20
            )
            btn.pack(fill=tk.X, pady=2)
            self.annotation_buttons.append(btn)
        optimize_button = ttk_boot.Button(
            self.button_frame,
            text="Finish & Optimize",
            command=self.finish_and_optimize,
            bootstyle="outline-danger",
            width=20
        )
        optimize_button.pack(fill=tk.X, pady=2)

        self.manage_button = ttk_boot.Button(
            self.button_frame,
            text="Manage keypoints",
            command=self.manage_existing_keypoints,
            bootstyle="outline-warning",
            width=20
        )
        self.manage_button.pack(fill=tk.X, pady=2)
        self.re_annotate_button = ttk_boot.Button(
            self.button_frame,
            text="Mutiview_2D_keypoint",
            command=self.add_multiview_keypoint,
            bootstyle="success",
            width=20
        )
        self.re_annotate_button.pack(fill=tk.X, pady=2)
        self.re_annotate_button.config(state=tk.DISABLED)


        
        # 添加物体缩放控制
        self.scale_frame = ttk_boot.Frame(self.plot_frame)
        self.scale_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.scale_label = ttk_boot.Label(
            self.scale_frame,
            text="Object Scale:",
            font=INFO_FONT,
            bootstyle=INFO
        )
        self.scale_label.pack(side=tk.LEFT)
        
        self.scale_var = tk.StringVar(value="1.0")
        self.scale_entry = ttk_boot.Entry(
            self.scale_frame,
            textvariable=self.scale_var,
            width=10,
            bootstyle=PRIMARY
        )
        self.scale_entry.pack(side=tk.LEFT, padx=(5, 5))
        
        self.scale_apply_btn = ttk_boot.Button(
            self.scale_frame,
            text="Apply",
            command=self.apply_object_scale,
            bootstyle="outline-primary",
            width=8
        )
        self.scale_apply_btn.pack(side=tk.LEFT)


        
        self.point_label = ttk_boot.Label(
            self.plot_frame, 
            text="No point selected", 
            font=INFO_FONT,
            bootstyle=INFO
        )
        self.point_label.pack(pady=(10, 5))
        self.info_frame = ttk_boot.Frame(self.plot_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        self.text_frame = ttk_boot.Frame(self.info_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.point_info = tk.Text(
            self.text_frame, 
            height=12, 
            width=35,
            font=(EMOJI_FONT, 9),
            bg="#2b3e50",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            selectbackground="#3498db",
            selectforeground="#ffffff",
            wrap=tk.WORD
        )
        
        self.scrollbar = ttk_boot.Scrollbar(
            self.text_frame, 
            orient=tk.VERTICAL, 
            command=self.point_info.yview,
            bootstyle=PRIMARY
        )
        self.point_info.config(yscrollcommand=self.scrollbar.set)
        
        self.point_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_human_panel(self):
        """设置人体关键点面板"""
        self.human_display_frame = ttk_boot.Frame(self.human_image_frame)
        self.human_display_frame.pack(fill=tk.BOTH, expand=True)

        self.human_status_label = ttk_boot.Label(
            self.human_image_frame,
            text="Click human keypoint to annotate",
            font=(EMOJI_FONT, 10, "italic"),
            bootstyle=INFO
        )
        self.human_status_label.pack(pady=(10, 0))
        
    def setup_render_panel(self):
        """设置渲染面板"""
        self.render_display_frame = ttk_boot.Frame(self.render_frame, width=500, height=500)
        self.render_display_frame.pack(fill=tk.BOTH, expand=True)
        self.render_display_frame.pack_propagate(False)

        self.render_status_label = ttk_boot.Label(
            self.render_display_frame,
            text="No annotation...",
            font=TITLE_FONT,
            bootstyle=WARNING
        )
        self.render_status_label.place(relx=0.5, rely=0.5, anchor="center")
        self.render_video_label = ttk_boot.Label(self.render_display_frame)
        self.render_video_label.place(x=0, y=0)
        
    def setup_bottom_images(self):
        """设置底部参考图像"""
        sides_pic = list(os.walk(resource_path("display")))[0][2]
        self.images_container = ttk_boot.Frame(self.bottom_frame)
        self.images_container.pack(fill=tk.X, pady=5)   
        for i, pic in enumerate(sides_pic):
            image_frame = ttk_boot.Frame(self.images_container, padding=5)
            image_frame.pack(side=tk.LEFT, padx=5)

            title_label = ttk_boot.Label(
                image_frame, 
                text=pic.split(".")[0], 
                font=INFO_FONT,
                bootstyle=SECONDARY
            )
            title_label.pack(pady=(0, 5))
            img = Image.open(os.path.join(resource_path("display"), pic))
            img = img.resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            setattr(self, pic.split(".")[0], photo)

            img_label = ttk_boot.Label(image_frame, image=photo, relief="solid", borderwidth=2)
            img_label.pack()
    
    def load_data(self, args):
        """加载数据"""
        self.video_dir = args.video_dir
        self.joint_to_optimize = args.joint_to_optimize
        self.cap = cv2.VideoCapture(f"{self.video_dir}/video.mp4")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frames)
        self.frame_change(0)
        if os.path.exists(f"{self.video_dir}/kp_record"):
            shutil.rmtree(f"{self.video_dir}/kp_record")
        os.makedirs(f"{self.video_dir}/kp_record")

        for frame in range(self.current_frame, self.total_frames):
            save_name = f"{frame}".zfill(5)
            with open(f"{args.video_dir}/kp_record/{save_name}.json", "w") as file:
                json.dump(self.kp_pair, file, indent=4)

        self.progress_bar.config(to=self.total_frames - 1)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            self.original_width, self.original_height = frame.size
            self.original_max_dim = max(self.original_height, self.original_width)
            self.keypoint_window_size = min(self.original_max_dim, 800)
            standard_size = (480, 480)
            frame = frame.resize(standard_size, Image.Resampling.LANCZOS)
            
            # 绘制追踪点（如果存在）
            frame = self.draw_tracking_points_on_frame(frame)
            
            self.img_width, self.img_height = frame.size
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label = ttk_boot.Label(self.video_display_frame, image=self.obj_img)
            self.video_label.pack(pady=10)
            self.root.geometry(f"{2*self.img_width + 360+560}x{self.img_height+500}")
            self.is_playing = False
            self.update_video_frame()

        # 为CoTracker准备视频数据
        if self.cotracker_model is not None and COTRACKER_AVAILABLE:
            try:
                video_path = f"{self.video_dir}/video.mp4"
                # 使用imageio加载视频帧，和demo保持一致
                self.video_frames = []
                for frame in iio.imiter(video_path, plugin="FFMPEG"):
                    self.video_frames.append(frame)
                print(f"Loaded {len(self.video_frames)} frames for CoTracker")
            except Exception as e:
                print(f"Failed to load video frames for CoTracker: {e}")
        
        # 初始化关键点缓存，加载现有数据（如果有的话）
        # self.load_keypoints_cache()
        # self.video_frames = None

        output = torch.load(f"{args.video_dir}/motion/result.pt")
        print(output.keys())
        self.body_params = output["smpl_params_incam"]
        self.global_body_params = output["smpl_params_global"]

        self.hand_poses = json.load(open(os.path.join(args.video_dir, 'motion/hand_pose.json')))
        self.human_part = json.load(open(f"{resource_path('video_optimizer/data/part_kp.json')}"))
        self.K = output['K_fullimg'][0]
        self.output = output
        self.R = torch.eye(3, dtype=torch.float32)
        self.T = torch.zeros(3, dtype=torch.float32)

        for i in range(self.total_frames):
            if str(i) not in self.hand_poses:
                self.hand_poses[str(i)] = {}
            self.body_params["body_pose"][i], self.hand_poses[str(i)]["left_hand"], self.hand_poses[str(i)]["right_hand"] \
                = update_hand_pose(self.hand_poses, self.body_params["global_orient"], self.body_params["body_pose"], i)

        with open(f'{args.video_dir}/output/obj_poses.json') as f:
            self.object_poses = json.load(f)
        
        self.obj_org = mesh = o3d.io.read_triangle_mesh(f"{args.video_dir}/obj_org.obj")
        self.sampled_obj = self.obj_org.simplify_quadric_decimation(target_number_of_triangles=1000)
        
        # self.obj_orgs, self.centers = preprocess_obj(self.obj_org, self.object_poses, os.path.join(args.video_dir, 'orient/'), self.total_frames)
        self.obj_orgs, self.t_finals = preprocess_obj(self.obj_org, self.object_poses, os.path.join(args.video_dir, 'orient/'), self.total_frames)
        self.sampled_orgs, _ = preprocess_obj(self.sampled_obj, self.object_poses, os.path.join(args.video_dir, 'orient/'), self.total_frames)
        
        self.R_finals = [np.eye(3)]*self.total_frames


        if "rotation" not in self.object_poses:
            self.object_poses['rotation'] = []
            for frame in range(self.current_frame, self.total_frames):
                self.object_poses['rotation'].append(np.eye(3))
        
        # self.body_pose_params = []
        # self.shape_params = []
        # self.left_hand_params = []
        # self.right_hand_params = []
        # self.global_orient = []
        # self.transl = []
        
        # for i in range(self.total_frames):
        #     self.body_pose_params.append(self.body_params["body_pose"][i].reshape(1, -1))
        #     self.shape_params.append(self.body_params['betas'][i].reshape(1, -1))
        #     handpose = self.hand_poses[str(i)]
        #     left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1,3)).float()
        #     right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1,3)).float()
        #     self.left_hand_params.append(left_hand_pose)
        #     self.right_hand_params.append(right_hand_pose)
        #     self.global_orient.append(self.body_params['global_orient'][i].reshape(1, 3))
        #     self.transl.append(self.body_params['transl'][i].reshape(1, -1))
        
        self.unwrapped_body_params()

        self.setup_human_keypoints()
        self.setup_bottom_images()
        self.update_frame_counter()
    def unwrapped_body_params(self):
        self.body_pose_params = []
        self.shape_params = []
        self.left_hand_params = []
        self.right_hand_params = []
        self.global_orient = []
        self.transl = []
        
        for i in range(self.total_frames):
            self.body_pose_params.append(self.body_params["body_pose"][i].reshape(1, -1))
            self.shape_params.append(self.body_params['betas'][i].reshape(1, -1))
            handpose = self.hand_poses[str(i)]
            left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1,3)).float()
            right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1,3)).float()
            self.left_hand_params.append(left_hand_pose)
            self.right_hand_params.append(right_hand_pose)
            self.global_orient.append(self.body_params['global_orient'][i].reshape(1, 3))
            self.transl.append(self.body_params['transl'][i].reshape(1, -1))
    def setup_human_keypoints(self):
        """设置人体关键点按钮"""
        obj_img_size = 480, 480
        img = Image.open(resource_path("human_kp.png"))
        human_img_width, human_img_height = img.size
        img = img.resize(obj_img_size, Image.Resampling.LANCZOS)
        self.human_img = ImageTk.PhotoImage(img)
        self.human_img_label = ttk_boot.Label(self.human_display_frame, image=self.human_img)
        self.human_img_label.pack(pady=10)
        
        # 添加关键点按钮
        for main_joint, location in self.main_joint_coord.items():
            real_x, real_y = location
            real_x, real_y = real_x - 40, real_y + 70
            if main_joint == "leftNeck":
                real_x, real_y = real_x + 25, real_y
            if main_joint == "rightNeck":
                real_x, real_y = real_x - 25, real_y
            
            scale_x = real_x / human_img_width * obj_img_size[0]
            scale_y = real_y / human_img_height * obj_img_size[1] - 10

            button = ttk_boot.Button(
                self.human_display_frame,
                text=self.button_name[main_joint],
                command=lambda x=main_joint, y_coord=scale_y, x_coord=scale_x: self.show_menu(x, x_coord, y_coord),
                bootstyle="outline-info",
                width=6
            )
            button.place(x=scale_x, y=scale_y)
    
    def toggle_video(self):
        """切换视频播放状态"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause", bootstyle="outline-warning")
        else:
            self.play_button.config(text="Play", bootstyle="success")
    
    def seek_video(self, value):
        """寻找视频帧"""
        was_playing = self.is_playing
        self.is_playing = False
        frame_no = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        self.frame_change(frame_no)
        # print(self.current_frame)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((480, 480), Image.Resampling.LANCZOS)
            
            # 绘制追踪点
            frame = self.draw_tracking_points_on_frame(frame)
            
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label.configure(image=self.obj_img)
            self.update_frame_counter()
        
        self.is_playing = was_playing
    
    def update_frame_counter(self):
        """更新帧计数器"""
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames - 1}")
    
    # def load_keypoints_cache(self):
    #     """加载现有的关键点数据到缓存中"""
    #     if not hasattr(self, 'video_dir') or not self.video_dir:
    #         return
            
    #     kp_record_dir = f"{self.video_dir}/kp_record"
    #     if not os.path.exists(kp_record_dir):
    #         return
            
    #     # 遍历所有帧文件，加载关键点数据到缓存
    #     for frame_idx in range(self.total_frames):
    #         save_name = f"{frame_idx}".zfill(5)
    #         kp_file = f"{kp_record_dir}/{save_name}.json"
            
    #         if os.path.exists(kp_file):
    #             try:
    #                 with open(kp_file, 'r') as f:
    #                     frame_data = json.load(f)
    #                     # 缓存该帧的关键点数据
    #                     self.frame_keypoints_cache[frame_idx] = frame_data
    #             except Exception as e:
    #                 print(f"Failed to load keypoints for frame {frame_idx}: {e}")
        
    #     print(f"Loaded keypoints cache for {len(self.frame_keypoints_cache)} frames")
    
    def draw_tracking_points_on_frame(self, frame):
        """在帧上绘制追踪点"""
        if not self.tracking_active or not self.tracked_points:
            return frame
        
        # 从缓存读取当前帧的2D关键点
        current_kp = self.frame_keypoints_cache.get(self.current_frame, {})
        # print('draw', current_kp)
        
        if "2D_keypoint" in current_kp and current_kp["2D_keypoint"]:
            try:
                # 转换PIL图像为numpy数组以便绘制
                frame_np = np.array(frame)
                
                # 计算缩放比例（原视频分辨率 vs 显示分辨率480x480）
                height, width = frame_np.shape[:2]
                # print(self.original_width, self.original_height)
                scale_x = width / self.original_width
                scale_y = height / self.original_height
                
                for obj_idx, img_point in current_kp["2D_keypoint"]:
                    # 应用缩放
                    x = int(img_point[0] * scale_x)
                    y = int(img_point[1] * scale_y)
                    
                    # 确保点在图像范围内
                    if 0 <= x < width and 0 <= y < height:
                        # 绘制圆点
                        cv2.circle(frame_np, (x, y), 6, (0, 255, 0), -1)  # 绿色实心圆
                        cv2.circle(frame_np, (x, y), 8, (255, 255, 255), 2)  # 白色边框
                        
                        # 添加对象索引标签
                        cv2.putText(frame_np, f"Obj{obj_idx}", (x+10, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 转换回PIL图像
                frame = Image.fromarray(frame_np)
            except Exception as e:
                print(f"Error drawing tracking points: {e}")
        
        return frame
    
    def update_video_frame(self):
        """更新视频帧"""
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_change(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((480, 480), Image.Resampling.LANCZOS)
                
                # 绘制追踪点
                frame = self.draw_tracking_points_on_frame(frame)
                
                self.obj_img = ImageTk.PhotoImage(image=frame)
                self.video_label.configure(image=self.obj_img)
                self.progress_var.set(self.current_frame)
                self.update_frame_counter()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_change(0)
                self.progress_var.set(0)
                self.update_frame_counter()
                self.is_playing = False
                self.play_button.config(text="Play", bootstyle="success")
        
        self.root.after(30, self.update_video_frame)
    
    def show_menu(self, main_joint, x, y):
        """显示关键点菜单"""
        menu = tk.Menu(self.human_image_frame, tearoff=0, bg="#2b3e50", fg="#ecf0f1", activebackground="#3498db")
        for sub_joint in self.joint_tree[main_joint]:
            menu.add_command(label=sub_joint, command=lambda sj=sub_joint: self.option_selected(sj))
        menu.add_separator()
        menu.add_command(label="Cancel", command=lambda: self.option_selected(None))
        menu.post(self.human_image_frame.winfo_rootx() + int(x), self.human_image_frame.winfo_rooty() + int(y))
    
    def option_selected(self, option):
        """选择关键点选项"""
        if option == "exit" or option is None:
            return
        self.selected_2d_point = None
        self.selected_human_kp = option
        self.human_status_label.config(text=f"Selected: {option}")
        print(f"Selected: {option}")
        self.update_plot()
    
    # def keypoint_2D(self):
    #     """选择2D关键点"""
    #     current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    #     ret, frame = self.cap.read()
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
    #     display_img = frame.copy()
    #     height, width = display_img.shape[:2]
    #     max_size = 800
    #     max_dim = max(height, width)
        
    #     if max_dim > max_size:
    #         scale = max_size / max_dim
    #         new_h, new_w = int(height * scale), int(width * scale)
    #         display_img = cv2.resize(display_img, (new_w, new_h))
    #     else:
    #         scale = 1
            
    #     cv2.namedWindow("2D keypoint selection")
        
    #     clicked_point = [None]
        
    #     def mouse_callback(event, x, y, flags, param):
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             clicked_point[0] = (x, y)
    #             cv2.circle(display_img, (x, y), 8, (0, 255, 0), -1)
    #             cv2.putText(display_img, f"({x}, {y})", (x+10, y), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #             cv2.imshow("2D keypoint selection", display_img)
        
    #     cv2.setMouseCallback("2D keypoint selection", mouse_callback)
    #     cv2.imshow("2D keypoint selection", display_img)
        
    #     while clicked_point[0] is None:
    #         cv2.waitKey(30)
    #         try:
    #             if cv2.getWindowProperty("2D keypoint selection", cv2.WND_PROP_VISIBLE) < 1:
    #                 break
    #         except cv2.error:
    #             break
        
    #     try:
    #         cv2.destroyWindow("2D keypoint selection")
    #     except cv2.error:
    #         pass
        
    #     if clicked_point[0] is not None:
    #         x, y = clicked_point[0]
    #         x /= scale
    #         y /= scale
    #         self.selected_2d_point = (x, y)
    #         self.update_plot()



    def _select_2d_point_on_frame(self, window_title="2D keypoint selection", image=None):
        """从图像中选择一个2D点. 如果没有提供图像，则从当前视频帧中读取."""
        if image is None:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            if not ret:
                messagebox.showerror("Error", "Could not read video frame.")
                return None
        else:
            # 渲染的图像是RGB格式的numpy数组, OpenCV需要BGR格式
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print(frame.shape)
        display_img = frame.copy()
        height, width = display_img.shape[:2]
        max_size = 800
        max_dim = max(height, width)
        
        scale = 1.0
        if max_dim > max_size:
            scale = max_size / max_dim
            new_h, new_w = int(height * scale), int(width * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
            
        cv2.namedWindow(window_title)
        
        clicked_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                img_copy = display_img.copy()
                cv2.circle(img_copy, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(img_copy, f"({x}, {y})", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window_title, img_copy)
        
        cv2.setMouseCallback(window_title, mouse_callback)
        cv2.imshow(window_title, display_img)
        
        while clicked_point[0] is None:
            if cv2.waitKey(30) & 0xFF == 27: # Allow exit with ESC
                 break
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cv2.destroyWindow(window_title)
        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            return (x, y)
        return None

    def keypoint_2D(self):
        """选择2D关键点"""
        point = self._select_2d_point_on_frame()
        if point:
            self.selected_2d_point = point
            self.update_plot()
    def get_body_points(self):
        body_pose = self.body_pose_params[self.current_frame]
        shape = self.shape_params[self.current_frame]
        global_orient = self.global_orient[self.current_frame]
        left_hand_pose = self.left_hand_params[self.current_frame]
        right_hand_pose = self.right_hand_params[self.current_frame]
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)
        transl = self.transl[self.current_frame]

        output = model(betas=shape,
                      body_pose=body_pose,
                      left_hand_pose=left_hand_pose,
                      right_hand_pose=right_hand_pose,
                      jaw_pose=zero_pose,
                      leye_pose=zero_pose,
                      reye_pose=zero_pose,
                      global_orient=global_orient,
                      expression=torch.zeros((1, 10)).float(),
                      transl=transl)
        print('transl', transl, 'global_orient', global_orient)
        return output.vertices[0]
    def get_object_points(self):
        verts = np.asarray(self.obj_orgs[self.current_frame].vertices, dtype=np.float32)
        R = self.R_finals[self.current_frame]
        t = self.t_finals[self.current_frame]
        print('obj_param', R, t)
        return np.matmul(verts, R.T) + t
    def render_angle_view(self):
        render_width, render_height = 800, 600
        render = o3d.visualization.rendering.OffscreenRenderer(render_width, render_height)
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])
        human_mat = o3d.visualization.rendering.MaterialRecord()
        human_mat.base_color = [0.7, 0.3, 0.3, 1.0]
        human_mat.shader = "defaultLit"
        
        object_mat = o3d.visualization.rendering.MaterialRecord()
        object_mat.base_color = [0.3, 0.5, 0.7, 1.0]
        object_mat.shader = "defaultLit"

        overts = self.get_object_points()
        hverts = self.get_body_points().cpu()
        incam_params = (self.global_orient[self.current_frame], self.transl[self.current_frame])
        global_params = (self.global_body_params["global_orient"][self.current_frame], self.global_body_params["transl"][self.current_frame])
        hverts, overts = transform_to_global(incam_params, global_params, hverts, overts)

        vertices = np.concatenate([overts, hverts], axis=0)
        camera_params = create_camera_for_object(vertices, image_width=render_width, image_height=render_height)
        K = camera_params['intrinsics']
        R = camera_params['rotation_matrix']
        T = camera_params['camera_position']
        scene_center = np.mean(vertices, axis=0).reshape(3, 1)

        R_np = R.cpu().numpy() if torch.is_tensor(R) else np.array(R)
        T_np = T.cpu().numpy() if torch.is_tensor(T) else np.array(T)
        if T_np.ndim == 1:
            T_np = T_np.reshape(3, 1)
        original_camera_pos = T_np

        camera_self_rotation = rotate_camera(-self.rotation_angle, "y")
        rotated_R = R_np @ camera_self_rotation
        
        world_rotation_matrix = rotate_camera(self.rotation_angle, "y")
        camera_relative = original_camera_pos - scene_center
        rotated_camera_relative = world_rotation_matrix @ camera_relative
        new_camera_pos = rotated_camera_relative + scene_center
        final_R = rotated_R
        final_T = new_camera_pos.flatten()
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = final_R
        extrinsic_matrix[:3, 3] = final_T
        view_matrix = np.linalg.inv(extrinsic_matrix)

        K_np = K.cpu().numpy() if torch.is_tensor(K) else np.array(K)
        cam_params = {
            'K': K_np.tolist(),
            'R': final_R.tolist(),
            'T': final_T.tolist(),
            'width': render_width,
            'height': render_height
        }
        self.current_multiview_cam_params = cam_params

        camera = o3d.camera.PinholeCameraIntrinsic(
            width=render_width,
            height=render_height,
            fx=K_np[0, 0],
            fy=K_np[1, 1],
            cx=K_np[0, 2],
            cy=K_np[1, 2]
        )

        human_faces = np.array(model.faces, dtype=np.int32)
        object_faces = np.asarray(self.obj_orgs[self.current_frame].triangles).astype(np.int32)
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(hverts)
        human_mesh.triangles = o3d.utility.Vector3iVector(human_faces)
        human_mesh.compute_vertex_normals()

        transformed_object_mesh = o3d.geometry.TriangleMesh()
        transformed_object_mesh.vertices = o3d.utility.Vector3dVector(overts)
        transformed_object_mesh.triangles = o3d.utility.Vector3iVector(object_faces)
        transformed_object_mesh.compute_vertex_normals()
        render.scene.add_geometry("human", human_mesh, human_mat)
        render.scene.add_geometry("object", transformed_object_mesh, object_mat)

        render.setup_camera(camera, view_matrix)
        rendered_img = render.render_to_image()

        return np.asarray(rendered_img), cam_params
        
    def add_multiview_keypoint(self):
        """为渲染过的帧添加多视角2D关键点"""
        if not self.modify_obj_point:
            messagebox.showwarning("操作无效", "请先使用 'Select 3D point' 按钮选择一个物体上的点。")
            return

        angle = simpledialog.askfloat("Input Angle",
                                      "Enter rotation angle (degrees):",
                                      parent=self.root,
                                      minvalue=-360.0,
                                      maxvalue=360.0,
                                      initialvalue=0.0) # Default to a common angle

        if angle is not None:
            self.rotation_angle = angle + 90.0
            angle_key = str(angle)

            # 1. 渲染新视角
            rendered_image, cam_params = self.render_angle_view()

            # 2. 在渲染出的图像上选择一个点
            point = self._select_2d_point_on_frame(
                window_title=f"Select point for angle {angle}",
                image=rendered_image
            )
            
            if point:
                # 3. 更新 self.kp_pair
                if angle_key not in self.kp_pair["multiview_2d_keypoints"]:
                    self.kp_pair["multiview_2d_keypoints"][angle_key] = {
                        "keypoints": [],
                        "cam_params": cam_params
                    }
                self.kp_pair["multiview_2d_keypoints"][angle_key]["keypoints"].append([int(self.modify_obj_point), point])

                # 4. 更新UI和标注状态
                self.point_info.insert(tk.END, f"Frame: {self.current_frame} | Angle: {angle}\n")
                self.point_info.insert(tk.END, f"  Object point: {self.modify_obj_point}\n")
                self.point_info.insert(tk.END, f"  Multiview 2D keypoint: {point}\n")
                self.point_info.see(tk.END)

                self.modify_annot = False
                self.annotated_modified_frames.add(self.current_frame)
                if len(self.annotated_modified_frames) > 1:
                    self.last_modified_frame = sorted(list(self.annotated_modified_frames))[-2]
                for frame_idx in range(self.current_frame, self.total_frames):
                    save_name = f"{self.video_dir}/kp_record/{str(frame_idx).zfill(5)}.json"
                    try:
                        with open(save_name, "r") as file:
                            kp_data = json.load(file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        kp_data = {"2D_keypoint": [], "multiview_2d_keypoints": {}}

                    kp_data["multiview_2d_keypoints"] = self.kp_pair["multiview_2d_keypoints"]

                    with open(save_name, "w") as file:
                        json.dump(kp_data, file, indent=4)

                # 6. 重置选择状态
                self.modify_obj_point = None
                self.point_label.config(text="No point selected")
    
    def track_2D_points_with_cotracker_online(self, obj_indices, start_points, start_frame=0):
        """使用CoTracker Online追踪2D点"""
        if not COTRACKER_AVAILABLE or self.cotracker_model is None or self.video_frames is None:
            print("CoTracker not available, falling back to interpolation")
            return False
        
        try:
            device = next(self.cotracker_model.parameters()).device
            
            # 准备查询点：[frame_idx, x, y]
            queries = []
            for i, (obj_idx, point) in enumerate(zip(obj_indices, start_points)):
                queries.append([float(start_frame), point[0], point[1]])
            print('queries',queries)
            
            queries_tensor = torch.tensor(queries, dtype=torch.float32).to(device)
            print(f"Tracking {len(queries)} points from frame {start_frame}")
            
            # 按照demo.py的方式进行在线追踪
            window_frames = [self.video_frames[start_frame]]  # 从起始帧开始

            def _process_step(window_frames, is_first_step, queries=None):
                video_chunk = (
                    torch.tensor(
                        np.stack(window_frames[-self.cotracker_model.step * 2 :]), device=device
                    )
                    .float()
                    .permute(0, 3, 1, 2)[None]
                )  # (1, T, 3, H, W)
                return self.cotracker_model(
                    video_chunk,
                    is_first_step=is_first_step,
                    queries=queries,
                    grid_size=0,  # 不使用grid，使用点坐标
                    grid_query_frame=0,
                )

            # 处理视频帧，完全按照demo.py的逻辑
            is_first_step = True
            for i in range(start_frame + 1, len(self.video_frames)):
                # 调整索引以适应跳过第一帧
                if (i - start_frame - 1) % self.cotracker_model.step == 0 and i != start_frame + 1:
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        queries=queries_tensor[None] if is_first_step else None,  # 只在第一步传入queries
                    )
                    is_first_step = False
                window_frames.append(self.video_frames[i])
            
            # 处理最后的视频帧
            i = len(self.video_frames) - 1  # 最后一帧的索引
            pred_tracks, pred_visibility = _process_step(
                window_frames[-(i % self.cotracker_model.step) - self.cotracker_model.step - 1 :],
                is_first_step,
                queries=queries_tensor[None] if is_first_step else None,
            )
            pred_tracks=pred_tracks[0].permute(1,0,2).cpu().numpy()  # [num_points, num_frames, 2]
            pred_visibility=pred_visibility[0].permute(1,0).cpu().numpy()
            
            for i, obj_idx in enumerate(obj_indices):
                tracks = pred_tracks[i]  # [num_frames, 2]
                visibility = pred_visibility[i]  # [num_frames]

                self.tracked_points[obj_idx] = []
                for frame_idx in range(len(tracks)):
                    # if frame_idx < len(visibility) and visibility[frame_idx] > 0.1:  # 可见性阈值
                    # print(visibility[frame_idx])
                    self.tracked_points[obj_idx].append(tracks[frame_idx].tolist())
                    # else:
                    #     self.tracked_points[obj_idx].append(None)  # 不可见
                
            print(f"CoTracker online tracking completed for {len(obj_indices)} points")
            return True

            
        except Exception as e:
            print(f"CoTracker online tracking failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_tracking_results_to_all_frames(self):
        """将追踪结果应用到所有帧"""
        if not self.tracked_points:
            return
        
        print("Applying tracking results to all frames...")
        
        for frame_idx in range(self.total_frames):
            frame_file = f"{self.video_dir}/kp_record/{str(frame_idx).zfill(5)}.json"
            
            # 读取现有数据
            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    current_kp = json.load(file)
            else:
                current_kp = {"2D_keypoint": []}
            
            # 更新2D关键点
            current_kp["2D_keypoint"] = []
            for obj_idx, tracks in self.tracked_points.items():
                if frame_idx < len(tracks) and tracks[frame_idx] is not None:
                    current_kp["2D_keypoint"].append([obj_idx, tracks[frame_idx]])
            
            # 保存到文件
            with open(frame_file, "w") as file:
                json.dump(current_kp, file, indent=4)
            
            # 同时更新缓存
            self.frame_keypoints_cache[frame_idx] = current_kp.copy()
        
        print("Tracking results applied to all frames")
        
        # 刷新当前显示的帧以显示追踪点
        self.refresh_current_frame()
    
    def apply_tracking_results_from_current_frame(self):
        """将追踪结果应用到从当前帧开始的所有帧"""
        if not self.tracked_points:
            return
        
        print(f"Applying tracking results from frame {self.current_frame} to end...")
        
        for frame_idx in range(self.current_frame, self.total_frames):
            frame_file = f"{self.video_dir}/kp_record/{str(frame_idx).zfill(5)}.json"
            
            # 读取现有数据
            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    current_kp = json.load(file)
            else:
                current_kp = {"2D_keypoint": []}
            # print(current_kp)
            
            # 更新2D关键点（保留3D关键点等其他数据）
            current_kp["2D_keypoint"] = []
            # print("Tracked points:", self.tracked_points)
            for obj_idx, tracks in self.tracked_points.items():
                # 计算在追踪结果中的索引（追踪从current_frame开始）
                track_idx = frame_idx - self.current_frame
                if track_idx < len(tracks) and tracks[track_idx] is not None:
                    current_kp["2D_keypoint"].append([obj_idx, tracks[track_idx]])
            # print(current_kp)
            
            # 保存到文件
            with open(frame_file, "w") as file:
                json.dump(current_kp, file, indent=4)
            
            # 同时更新缓存
            self.frame_keypoints_cache[frame_idx] = current_kp.copy()
        
        print(f"Tracking results applied from frame {self.current_frame} to end")
        
        # 刷新当前显示的帧以显示追踪点
        self.refresh_current_frame()
    
    def refresh_current_frame(self):
        """刷新当前显示的帧"""
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((480, 480), Image.Resampling.LANCZOS)
            
            # 绘制追踪点
            frame = self.draw_tracking_points_on_frame(frame)
            
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label.configure(image=self.obj_img)
    
    def manage_existing_keypoints(self):
        """管理现有关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if not os.path.exists(current_file):
            messagebox.showinfo("Info", "No keypoint annotation")
            return
        
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        is_rendered = self.current_frame in self.rendered_frames

        # 创建管理窗口
        manage_window = ttk_boot.Toplevel(self.root)
        manage_window.title(f"Manage keypoints - Frame {self.current_frame}")
        manage_window.geometry("900x600")

        style = ttk_boot.Style(theme="superhero")
        main_frame = ttk_boot.Frame(manage_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = ttk_boot.Label(
            main_frame, 
            text=f"Frame {self.current_frame} keypoints", 
            font=TITLE_FONT,
            bootstyle=PRIMARY
        )
        title_label.pack(pady=(0, 20))
        canvas = tk.Canvas(main_frame, bg="#2b3e50", highlightthickness=0)
        scrollbar = ttk_boot.Scrollbar(main_frame, orient="vertical", command=canvas.yview, bootstyle=PRIMARY)
        scrollable_frame = ttk_boot.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        has_any_keypoint = False

        if not is_rendered:
            if "2D_keypoint" in current_kp and current_kp["2D_keypoint"]:
                has_any_keypoint = True
                section_frame = ttk_boot.LabelFrame(scrollable_frame, text="2D keypoint", padding=15, bootstyle=INFO)
                section_frame.pack(fill=tk.X, pady=(0, 10))
                
                for i, (obj_idx, img_point) in enumerate(current_kp["2D_keypoint"]):
                    item_frame = ttk_boot.Frame(section_frame)
                    item_frame.pack(fill=tk.X, pady=5)
                    info_label = ttk_boot.Label(
                        item_frame, 
                        text=f"Object point {obj_idx} -> Image point ({img_point[0]:.1f}, {img_point[1]:.1f})",
                        font=DEFAULT_FONT
                    )
                    info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    btn_frame = ttk_boot.Frame(item_frame)
                    btn_frame.pack(side=tk.RIGHT)
                    modify_btn = ttk_boot.Button(
                        btn_frame, 
                        text="Modify", 
                        command=lambda idx=i: self.modify_2d_keypoint(idx, manage_window),
                        bootstyle=WARNING,
                        width=10
                    )
                    modify_btn.pack(side=tk.LEFT, padx=2)
                    delete_btn = ttk_boot.Button(
                        btn_frame, 
                        text="Delete", 
                        command=lambda idx=i: self.delete_2d_keypoint(idx, manage_window),
                        bootstyle=DANGER,
                        width=10
                    )
                    delete_btn.pack(side=tk.LEFT, padx=2)
            
            has_3d_keypoints = False
            for key, value in current_kp.items():
                if key not in ["2D_keypoint", "multiview_2d_keypoints", "multiview_cam_params"]:
                    if not has_3d_keypoints:
                        has_any_keypoint = True
                        section_frame = ttk_boot.LabelFrame(scrollable_frame, text="3D keypoint", padding=15, bootstyle=SUCCESS)
                        section_frame.pack(fill=tk.X, pady=(0, 10))
                        has_3d_keypoints = True
                    
                    item_frame = ttk_boot.Frame(section_frame)
                    item_frame.pack(fill=tk.X, pady=5)
                    joint_name = self.button_name.get(key, key)
                    info_label = ttk_boot.Label(
                        item_frame, 
                        text=f"{joint_name} -> Object point {value}",
                        font=DEFAULT_FONT
                    )
                    info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    btn_frame = ttk_boot.Frame(item_frame)
                    btn_frame.pack(side=tk.RIGHT)
                    modify_btn = ttk_boot.Button(
                        btn_frame, 
                        text="Modify", 
                        command=lambda joint=key: self.modify_3d_keypoint(joint, manage_window),
                        bootstyle=WARNING,
                        width=10
                    )
                    modify_btn.pack(side=tk.LEFT, padx=2)

                    delete_btn = ttk_boot.Button(
                        btn_frame, 
                        text="Delete", 
                        command=lambda joint=key: self.delete_3d_keypoint(joint, manage_window),
                        bootstyle=DANGER,
                        width=10
                    )
                    delete_btn.pack(side=tk.LEFT, padx=2)
            
        if "multiview_2d_keypoints" in current_kp and current_kp["multiview_2d_keypoints"]:
            has_any_keypoint = True
            multiview_frame = ttk_boot.LabelFrame(scrollable_frame, text="Multiview 2D Keypoints", padding=15, bootstyle=SUCCESS)
            multiview_frame.pack(fill=tk.X, pady=(0, 10))

            for angle, data in current_kp["multiview_2d_keypoints"].items():
                angle_label = ttk_boot.Label(multiview_frame, text=f"Angle: {angle}°", font=SUBTITLE_FONT, bootstyle=SECONDARY)
                angle_label.pack(fill=tk.X, pady=(10, 5))

                for i, (obj_idx, img_point) in enumerate(data.get("keypoints", [])):
                    item_frame = ttk_boot.Frame(multiview_frame)
                    item_frame.pack(fill=tk.X, pady=5)
                    info_label = ttk_boot.Label(
                        item_frame,
                        text=f"  Object point {obj_idx} -> Image point ({img_point[0]:.1f}, {img_point[1]:.1f})",
                        font=DEFAULT_FONT
                    )
                    info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    btn_frame = ttk_boot.Frame(item_frame)
                    btn_frame.pack(side=tk.RIGHT)
                    modify_btn = ttk_boot.Button(
                        btn_frame,
                        text="Modify",
                        command=lambda ang=angle, idx=i: self.modify_multiview_keypoint(ang, idx, manage_window),
                        bootstyle=WARNING,
                        width=10
                    )
                    modify_btn.pack(side=tk.LEFT, padx=2)
                    delete_btn = ttk_boot.Button(
                        btn_frame,
                        text="Delete",
                        command=lambda ang=angle, idx=i: self.delete_multiview_keypoint(ang, idx, manage_window),
                        bootstyle=DANGER,
                        width=10
                    )
                    delete_btn.pack(side=tk.LEFT, padx=2)

        if not has_any_keypoint:
            empty_label = ttk_boot.Label(
                scrollable_frame, 
                text="No keypoint annotation", 
                font=SUBTITLE_FONT,
                bootstyle=WARNING
            )
            empty_label.pack(pady=50)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        close_btn = ttk_boot.Button(
            main_frame, 
            text="Close", 
            command=manage_window.destroy,
            bootstyle=SUCCESS,
            width=15
        )
        close_btn.pack(pady=(20, 0))
    def delete_multiview_keypoint(self, angle_key, index, parent_window):
        """删除多视角2D关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if "multiview_2d_keypoints" in current_kp and \
           angle_key in current_kp["multiview_2d_keypoints"] and \
           index < len(current_kp["multiview_2d_keypoints"][angle_key]["keypoints"]):
            
            del current_kp["multiview_2d_keypoints"][angle_key]["keypoints"][index]

            # 如果这个角度下没有关键点了，把整个角度条目删掉
            if not current_kp["multiview_2d_keypoints"][angle_key]["keypoints"]:
                del current_kp["multiview_2d_keypoints"][angle_key]

            self.kp_pair = current_kp
            self.modify_annot = False 
            self.annotated_modified_frames.add(self.current_frame)
            if len(self.annotated_modified_frames) > 1:
                self.last_modified_frame = sorted(list(self.annotated_modified_frames))[-2]
            
            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                # Ensure the file exists before writing, or create a default structure
                try:
                    with open(save_name, 'r') as f:
                        kp_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    kp_data = {"2D_keypoint": [], "multiview_2d_keypoints": {}}
                
                kp_data['multiview_2d_keypoints'] = current_kp['multiview_2d_keypoints']
                with open(save_name, "w") as file:
                    json.dump(kp_data, file, indent=4)
            
            parent_window.destroy()
            messagebox.showinfo("Success", "Multiview 2D keypoint deleted")
            self.manage_existing_keypoints()
    
    def reset_keypoints(self):
        """重置关键点"""
        self.kp_pair = {"2D_keypoint": []}
        self.tracked_points = {}
        self.tracking_active = False
        self.point_info.delete('1.0', tk.END)
        self.point_info.insert(tk.END, "Keypoints reset\n")
        self.point_label.config(text="No point selected")
        self.human_status_label.config(text="Click human keypoint to annotate")
        self.refresh_current_frame()
        with open(f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json", "w") as file:
            json.dump(self.kp_pair, file, indent=4)
    
    def apply_object_scale(self):
        """应用物体缩放"""
        try:
            scale_factor = float(self.scale_var.get())
            if scale_factor <= 0:
                messagebox.showerror("错误", "缩放因子必须大于0")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            return
        
        # 确认对话框
        result = messagebox.askyesno("确认缩放", 
                                   f"将对所有帧的物体应用 {scale_factor}x 缩放。\n"
                                   f"这个操作会修改物体的几何形状。\n\n确定要继续吗？")
        
        if not result:
            return
        
        # 应用缩放到所有帧
        for frame_idx in range(self.total_frames):
            if frame_idx < len(self.obj_orgs):
                # 获取当前帧的物体
                obj = self.obj_orgs[frame_idx]

                # 计算物体的中心点（原点）
                vertices = np.asarray(obj.vertices)
                
                # 平移到原点（相对于center）
                center=np.mean(vertices, axis=0)
                vertices_centered = vertices - center

                # 应用缩放
                vertices_scaled = vertices_centered * scale_factor
                
                # 平移回原位置
                vertices_final = vertices_scaled + center
                
                # 更新物体的顶点
                obj.vertices = o3d.utility.Vector3dVector(vertices_final)
                
                # 重新计算法向量
                # obj.compute_vertex_normals()
        
        # 同样处理采样后的物体
        for frame_idx in range(self.total_frames):
            if frame_idx < len(self.sampled_orgs):
                # 获取当前帧的采样物体
                obj = self.sampled_orgs[frame_idx]
                
                # 计算物体的中心点（原点）
                vertices = np.asarray(obj.vertices)
                
                # 平移到原点（相对于center）
                center=np.mean(vertices, axis=0)
                vertices_centered = vertices - center
                
                # 应用缩放
                vertices_scaled = vertices_centered * scale_factor
                
                # 平移回原位置
                vertices_final = vertices_scaled + center
                
                # 更新物体的顶点
                obj.vertices = o3d.utility.Vector3dVector(vertices_final)
                
                # 重新计算法向量
                # obj.compute_vertex_normals()
        
        # 刷新当前显示的帧
        self.refresh_current_frame()
        
        messagebox.showinfo("完成", f"物体缩放 {scale_factor}x 应用完成！")
        self.point_info.insert(tk.END, f"Applied {scale_factor}x scale to all frames\n")
        self.point_info.see(tk.END)
    
    def finish_and_optimize(self):
        """结束标注并优化最后一段视频"""
        # 检查是否有标注数据
        if not self.annotated_frames or len(self.annotated_frames) < 1:
            messagebox.showwarning("Warning", "没有标注数据，无法进行优化")
            return
        if not self.render_key_frame:
            start_frame = 0
        else:
            start_frame = sorted(self.render_key_frame)[-1]
        end_frame = self.current_frame
        self.render_key_frame.add(end_frame)
        
        # 确认对话框
        result = messagebox.askyesno("确认优化", 
                                   f"将对第 {start_frame} 帧到第 {end_frame} 帧的段落进行优化。\n"
                                   f"这个过程可能需要一些时间。\n\n确定要开始优化吗？")
        
        if not result:
            return
        
        self.render_status_label.config(text="Starting optimization...")
        
        # 在后台线程中运行优化
        def run_optimization():
            # 设置状态开始优化
            self.no_annot = False
            
            # 运行优化
            self.show_video_interval(start_frame, end_frame)
            
            # 优化完成后设置状态
            self.no_annot = True
            
            # 在UI线程中安全地保存最终参数
            self.root.after(0, self.save_final_optimized_parameters)
            self.root.after(0, lambda: messagebox.showinfo("完成", f"段落 {start_frame}-{end_frame} 优化完成！\n参数已自动保存。"))
        threading.Thread(target=run_optimization, daemon=True).start()
    
    def start_2d_tracking(self):
        """手动启动2D点追踪"""
        if not self.kp_pair["2D_keypoint"]:
            messagebox.showwarning("Warning", "请先标注2D关键点才能开始追踪")
            return
        
        if self.tracking_active:
            messagebox.showinfo("Info", "追踪已经在进行中")
            return
        
        # 确认对话框
        result = messagebox.askyesno("确认追踪", 
                                   f"将从第 {self.current_frame} 帧开始追踪 {len(self.kp_pair['2D_keypoint'])} 个2D点。\n"
                                   f"这将覆盖所有帧的2D关键点数据。\n\n确定要开始追踪吗？")
        
        if not result:
            return
        
        self.render_status_label.config(text="Starting tracking...")
        
        # 提取所有对象索引和起始点
        obj_indices = [pair[0] for pair in self.kp_pair["2D_keypoint"]]
        start_points = [pair[1] for pair in self.kp_pair["2D_keypoint"]]
        
        # 启动追踪（在后台线程中运行）
        def run_tracking():
            success = self.track_2D_points_with_cotracker_online(obj_indices, start_points, self.current_frame)
            if success:
                self.tracking_active = True
                # 将追踪结果应用到所有帧
                self.apply_tracking_results_to_all_frames()
                self.root.after(0, lambda: self.render_status_label.config(text="Tracking completed"))
                self.root.after(0, lambda: messagebox.showinfo("成功", "2D点追踪完成！"))
            else:
                self.root.after(0, lambda: self.render_status_label.config(text="Tracking failed"))
                self.root.after(0, lambda: messagebox.showerror("错误", "追踪失败，请检查CoTracker是否正确安装"))
        
        threading.Thread(target=run_tracking, daemon=True).start()
    
    def restart_tracking_from_current(self):
        """从当前帧重新开始追踪"""
        # 检查当前帧是否有2D关键点
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if not os.path.exists(current_file):
            messagebox.showwarning("Warning", "当前帧没有保存的关键点数据")
            return
        
        # 读取当前帧的2D关键点
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if "2D_keypoint" not in current_kp or not current_kp["2D_keypoint"]:
            messagebox.showwarning("Warning", "当前帧没有2D关键点标注")
            return
        
        # 确认对话框
        result = messagebox.askyesno("确认重新追踪", 
                                   f"将从第 {self.current_frame} 帧重新开始追踪 {len(current_kp['2D_keypoint'])} 个2D点。\n"
                                   f"这将重写从当前帧到最后一帧的所有2D关键点数据。\n\n确定要重新追踪吗？")
        
        if not result:
            return
        
        self.render_status_label.config(text="Re-tracking from current frame...")
        
        # 提取当前帧的对象索引和起始点
        print('rr',current_kp)
        obj_indices = [pair[0] for pair in current_kp["2D_keypoint"]]
        start_points = [pair[1] for pair in current_kp["2D_keypoint"]]
        
        # 启动重新追踪（在后台线程中运行）
        def run_retracking():
            success = self.track_2D_points_with_cotracker_online(obj_indices, start_points, self.current_frame)
            if success:
                self.tracking_active = True
                # 将追踪结果应用到从当前帧开始的所有帧
                self.apply_tracking_results_from_current_frame()
                self.root.after(0, lambda: self.render_status_label.config(text="Re-tracking completed"))
                self.root.after(0, lambda: messagebox.showinfo("成功", f"从第 {self.current_frame} 帧重新追踪完成！"))
            else:
                self.root.after(0, lambda: self.render_status_label.config(text="Re-tracking failed"))
                self.root.after(0, lambda: messagebox.showerror("错误", "重新追踪失败，请检查CoTracker是否正确安装"))
        
        threading.Thread(target=run_retracking, daemon=True).start()
    
    def open_o3d_viewer(self):
        """打开3D查看器"""
        vertices = self.get_object_points()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = self.obj_orgs[self.current_frame].triangles
        mesh.vertex_colors = self.obj_orgs[self.current_frame].vertex_colors
        mesh.compute_vertex_normals()
        human_points = self.get_body_points()

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
        # elif mesh.has_triangles() and mesh.has_vertex_normals():
        #     normals = np.asarray(mesh.vertex_normals)
        #     colors = (normals + 1) / 2
        #     pcd.colors = o3d.utility.Vector3dVector(colors)
        # else:
        #     vertices = np.asarray(mesh.vertices)
        #     min_bound = vertices.min(axis=0)
        #     max_bound = vertices.max(axis=0)
        #     norm_vertices = (vertices - min_bound) / (max_bound - min_bound)
        #     colors = np.zeros((len(vertices), 3))
        #     colors[:, 0] = norm_vertices[:, 0]
        #     colors[:, 1] = norm_vertices[:, 1]
        #     colors[:, 2] = norm_vertices[:, 2]
        #     pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(human_points)
        pcd2.colors = o3d.utility.Vector3dVector(human_points * 0.5 + 0.5)

        combined_pcd = pcd + pcd2

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="3D Viewer")
        vis.add_geometry(combined_pcd)
        opt = vis.get_render_option()
        opt.point_size = 10.0
        vis.run()

        picked_points = vis.get_picked_points()
        if picked_points:
            picked_idx = picked_points[-1]
            if picked_idx >= len(pcd.points):
                picked_idx -= len(pcd.points)
                picked_point = np.asarray(pcd2.points)[picked_idx]
                obj_points = np.asarray(pcd.points)
                distances = np.linalg.norm(obj_points - picked_point, axis=1)
                nearest_idx = np.argmin(distances)
                if self.current_frame in self.rendered_frames:
                    self.modify_obj_point = str(nearest_idx)
                else:
                    self.obj_point = str(nearest_idx)
            else:
                if self.current_frame in self.rendered_frames:
                    self.modify_obj_point = str(picked_idx)
                else:
                    self.obj_point = str(picked_idx)
            if self.current_frame in self.rendered_frames:
                self.point_label.config(text=f"Selected point for modification: {self.modify_obj_point}")
            else:
                self.point_label.config(text=f"Selected point: {self.obj_point}")
        else:
            self.obj_point = None
            self.point_label.config(text="No point selected")
        vis.destroy_window()
    
    def update_plot(self):
        # update kp_pair
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if os.path.exists(current_file):
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            self.kp_pair = current_kp

        """更新图表"""
        if self.selected_2d_point is not None and self.obj_point is not None:
            self.kp_pair["2D_keypoint"].append((int(self.obj_point), self.selected_2d_point))
            self.point_info.insert(tk.END, f"2D keypoint: {self.selected_2d_point}\n")
            self.point_info.insert(tk.END, f"Object point index: {self.obj_point}\n")
            self.point_info.insert(tk.END, f"Current frame: {self.current_frame}\n")
            self.point_info.insert(tk.END, "─" * 30 + "\n")
            self.point_info.see(tk.END)
            
            self.no_annot = False
            self.annotated_frames.add(self.current_frame)
            self.annotated_frames_2D.add(self.current_frame)
            if len(self.annotated_frames) > 1:
                self.last_frame = sorted(list(self.annotated_frames))[-2]
            else:
                self.last_frame = None
            
            if len(self.annotated_frames_2D) > 1:
                self.last_frame_2D = sorted(list(self.annotated_frames_2D))[-2]
            else:
                self.last_frame_2D = None

            # 如果追踪还未激活，且有2D关键点，提示用户手动启动追踪
            if not self.tracking_active and len(self.kp_pair["2D_keypoint"]) > 0:
                self.render_status_label.config(text="Click 'Start tracking' button to track points")
                # 只保存当前帧数据
                save_name = f"{self.current_frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                # 更新缓存
                self.frame_keypoints_cache[self.current_frame] = self.kp_pair.copy()
            else:
                # 如果追踪已激活，保存当前帧数据并询问是否重新追踪

                print('r',self.kp_pair)
                
                save_name = f"{self.current_frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                # 更新缓存
                self.frame_keypoints_cache[self.current_frame] = self.kp_pair.copy()

                # 刷新当前帧显示
                self.refresh_current_frame()
                
            return
        
        # 处理3D关键点
        if self.selected_human_kp is None or self.obj_point is None:
            return
        
        self.no_annot = False
        self.no_annote_2D = True
        self.annotated_frames.add(self.current_frame)
        
        if len(self.annotated_frames) > 1:
            self.last_frame = sorted(list(self.annotated_frames))[-2]
        else:
            self.last_frame = None
        
        self.kp_pair[self.selected_human_kp] = int(self.obj_point)
        
        # 更新信息显示
        self.point_info.insert(tk.END, f"Current frame: {self.current_frame}\n")
        self.point_info.insert(tk.END, f"Object point index: {self.obj_point}\n")
        self.point_info.insert(tk.END, f"Human keypoint: {self.selected_human_kp}\n")
        self.point_info.insert(tk.END, "─" * 30 + "\n")
        self.point_info.see(tk.END)
        
        # 保存到文件
        for frame in range(self.current_frame, self.total_frames):
            save_name = f"{frame}".zfill(5)
            frame_file = f"{self.video_dir}/kp_record/{save_name}.json"
            
            # 检查文件是否存在，如果存在则保留原有数据
            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    existing_data = json.load(file)
                # 保留原有的2D_keypoint数据，只更新3D关键点
                if "2D_keypoint" in existing_data:
                    self.kp_pair["2D_keypoint"] = existing_data["2D_keypoint"]
                # 保留其他可能存在的数据
                # for key, value in existing_data.items():
                #     if key not in [self.selected_human_kp]:
                #         if key not in self.kp_pair or key == "2D_keypoint":
                #             self.kp_pair[key] = value
            
            with open(frame_file, "w") as file:
                json.dump(self.kp_pair, file, indent=4)
        
        # 重置选择
        self.point_label.config(text="No point selected")
        self.human_status_label.config(text="Click human keypoint to annotate")
        self.render_status_label.config(text="Processing...")
    
    # 保持原有的其他方法...
    def delete_2d_keypoint(self, index, parent_window):
        """删除2D关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if "2D_keypoint" in current_kp and index < len(current_kp["2D_keypoint"]):
            # del current_kp["2D_keypoint"][index]
            # self.kp_pair = current_kp
            # self.no_annot = False
            # self.annotated_frames.add(self.current_frame)
            
            # if len(self.annotated_frames) > 1:
            #     self.last_frame = sorted(list(self.annotated_frames))[-2]
            # else:
            #     self.last_frame = None
            
            for frame in range(self.current_frame, self.total_frames):
                # current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
                current_file = f"{self.video_dir}/kp_record/{str(frame).zfill(5)}.json"
                with open(current_file, "r") as file:
                    current_kp = json.load(file)
                del current_kp["2D_keypoint"][index]
                self.kp_pair = current_kp
                print('kp_pair',self.kp_pair)


                save_name = f"{frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                # 更新缓存
                self.frame_keypoints_cache[frame] = self.kp_pair.copy()

            # 刷新当前帧显示
            self.refresh_current_frame()

            parent_window.destroy()
            messagebox.showinfo("Success", "2D keypoint deleted")


            self.manage_existing_keypoints()
    
    def delete_3d_keypoint(self, joint_name, parent_window):
        """删除3D关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if joint_name in current_kp:
            del current_kp[joint_name]
            self.kp_pair = current_kp
            self.no_annot = False
            self.annotated_frames.add(self.current_frame)
            
            if len(self.annotated_frames) > 1:
                self.last_frame = sorted(list(self.annotated_frames))[-2]
            else:
                self.last_frame = None
            
            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
            
            parent_window.destroy()
            messagebox.showinfo("Success", "3D keypoint deleted")
            self.manage_existing_keypoints()
    
    def modify_2d_keypoint(self, index, parent_window):
        """修改2D关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if "2D_keypoint" not in current_kp or index >= len(current_kp["2D_keypoint"]):
            return
        
        obj_idx, old_img_point = current_kp["2D_keypoint"][index]
        parent_window.destroy()
        self._select_new_2d(obj_idx, old_img_point, index)
        self.manage_existing_keypoints()
    
    def _select_new_2d(self, obj_idx, old_img_point, kp_index):
        """选择新的2D点"""
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        display_img = frame.copy()
        height, width = display_img.shape[:2]
        max_size = 800
        max_dim = max(height, width)
        
        if max_dim > max_size:
            scale = max_size / max_dim
            new_h, new_w = int(height * scale), int(width * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        else:
            scale = 1
        old_x, old_y = int(old_img_point[0] * scale), int(old_img_point[1] * scale)
        cv2.circle(display_img, (old_x, old_y), 8, (0, 0, 255), 2)
        cv2.putText(display_img, f"Old: Obj{obj_idx}", (old_x+10, old_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.namedWindow("2D keypoint modify")
        
        clicked_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                temp_img = display_img.copy()
                cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(temp_img, f"New: ({x}, {y})", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("2D keypoint modify", temp_img)
        
        cv2.setMouseCallback("2D keypoint modify", mouse_callback)
        cv2.imshow("2D keypoint modify", display_img)
        
        while clicked_point[0] is None:
            cv2.waitKey(30)
            try:
                if cv2.getWindowProperty("2D keypoint modify", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        
        try:
            cv2.destroyWindow("2D keypoint modify")
        except cv2.error:
            pass
        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            
            current_kp["2D_keypoint"][kp_index] = [obj_idx, [x, y]]
            self.kp_pair = current_kp
            self.no_annot = False
            self.no_annote_2D = False
            self.annotated_frames.add(self.current_frame)
            self.annotated_frames_2D.add(self.current_frame)
            
            if len(self.annotated_frames) > 1:
                self.last_frame = sorted(list(self.annotated_frames))[-2]
            else:
                self.last_frame = None
            if len(self.annotated_frames_2D) > 1:
                self.last_frame_2D = sorted(list(self.annotated_frames_2D))[-2]
            else:
                self.last_frame_2D = None
            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
                # 更新缓存
                self.frame_keypoints_cache[frame] = self.kp_pair.copy()
            
            # 刷新当前帧显示
            self.refresh_current_frame()
            
            messagebox.showinfo("Success", "2D keypoint modified")
    def modify_multiview_keypoint(self, angle_key, index, parent_window):
        """修改多视角2D关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
        if "multiview_2d_keypoints" not in current_kp or \
           angle_key not in current_kp["multiview_2d_keypoints"] or \
           index >= len(current_kp["multiview_2d_keypoints"][angle_key].get("keypoints", [])):
            return
        
        obj_idx, old_point = current_kp["multiview_2d_keypoints"][angle_key]["keypoints"][index]
        angle = float(angle_key)
        
        parent_window.destroy()

        # 重新渲染该视角
        original_rotation_angle = self.rotation_angle
        self.rotation_angle = angle + 90.0
        rendered_image, _ = self.render_angle_view()
        self.rotation_angle = original_rotation_angle
        
        # 调用新的选择函数
        self._select_new_multiview_2d(
            obj_idx,
            old_point,
            angle_key,
            index,
            rendered_image,
            window_title=f"Modify point for angle {angle}"
        )
        
        self.manage_existing_keypoints()
    def _select_new_multiview_2d(self, obj_idx, old_point, angle_key, kp_index, image, window_title):
        """为多视角标注选择新的2D点"""
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        display_img = frame.copy()
        
        height, width = display_img.shape[:2]
        scale = 1.0 # The image is already at the correct display size.

        old_x, old_y = int(old_point[0] * scale), int(old_point[1] * scale)
        cv2.circle(display_img, (old_x, old_y), 8, (0, 0, 255), 2)
        cv2.putText(display_img, f"Old: Obj{obj_idx}", (old_x + 10, old_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.namedWindow(window_title)
        
        clicked_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                temp_img = display_img.copy()
                cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(temp_img, f"New: ({x}, {y})", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_title, temp_img)
        
        cv2.setMouseCallback(window_title, mouse_callback)
        cv2.imshow(window_title, display_img)

        while clicked_point[0] is None:
            if cv2.waitKey(30) & 0xFF == 27:
                break
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass

        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            new_point = [x, y]
            
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            
            current_kp["multiview_2d_keypoints"][angle_key]["keypoints"][kp_index] = [obj_idx, new_point]
            self.kp_pair = current_kp
            self.modify_annot = False
            self.annotated_modified_frames.add(self.current_frame)
            if len(self.annotated_modified_frames) > 1:
                self.last_modified_frame = sorted(list(self.annotated_modified_frames))[-2]

            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                # Ensure the file exists before writing
                try:
                    with open(save_name, 'r') as f:
                        kp_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    kp_data = {"2D_keypoint": [], "multiview_2d_keypoints": {}}
                
                kp_data['multiview_2d_keypoints'] = current_kp['multiview_2d_keypoints']
                with open(save_name, "w") as file:
                    json.dump(kp_data, file, indent=4)
            
            messagebox.showinfo("Success", "Multiview 2D keypoint modified")

    
    def modify_3d_keypoint(self, joint_name, parent_window):
        """修改3D关键点"""
        parent_window.destroy()
        joint_display_name = self.button_name.get(joint_name, joint_name)
        self.open_o3d_viewer()
        
        if self.obj_point is not None:
            current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
            with open(current_file, "r") as file:
                current_kp = json.load(file)
            
            current_kp[joint_name] = int(self.obj_point)
            self.kp_pair = current_kp
            self.no_annot = False
            
            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
            
            messagebox.showinfo("Success", "3D keypoint modified")
        self.manage_existing_keypoints()
    
    def track_2D_points_with_cotracker(self, obj_indices, start_points, start_frame=0):
        """使用CoTracker追踪2D点"""
        if not COTRACKER_AVAILABLE or self.cotracker_model is None or self.video_tensor is None:
            print("CoTracker not available, falling back to interpolation")
            return False
        
        try:
            # 准备查询点：[frame_idx, x, y]
            queries = []
            for i, (obj_idx, point) in enumerate(zip(obj_indices, start_points)):
                queries.append([float(start_frame), point[0], point[1]])
            
            queries_tensor = torch.tensor(queries, dtype=torch.float32).unsqueeze(0)
            device = next(self.cotracker_model.parameters()).device
            queries_tensor = queries_tensor.to(device)
            
            print(f"Tracking {len(queries)} points from frame {start_frame}")
            
            # 运行追踪器
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker_model(self.video_tensor, queries=queries_tensor)
            
            # 处理追踪结果
            pred_tracks = pred_tracks.squeeze(0).cpu().numpy()  # [num_points, num_frames, 2]
            pred_visibility = pred_visibility.squeeze(0).cpu().numpy()  # [num_points, num_frames]
            
            # 存储追踪结果
            for i, obj_idx in enumerate(obj_indices):
                tracks = pred_tracks[i]  # [num_frames, 2]
                visibility = pred_visibility[i]  # [num_frames]
                
                self.tracked_points[obj_idx] = []
                for frame_idx in range(len(tracks)):
                    if visibility[frame_idx] > 0.5:  # 可见性阈值
                        self.tracked_points[obj_idx].append(tracks[frame_idx].tolist())
                    else:
                        self.tracked_points[obj_idx].append(None)  # 不可见
            
            print(f"CoTracker tracking completed for {len(obj_indices)} points")
            return True
            
        except Exception as e:
            print(f"CoTracker tracking failed: {e}")
            return False
    

    
    def interpolate_2D(self, start_frame, end_frame):
        """2D关键点插值"""
        start_file = f"{self.video_dir}/kp_record/{str(start_frame).zfill(5)}.json"
        end_file = f"{self.video_dir}/kp_record/{str(end_frame).zfill(5)}.json"
        
        with open(start_file, "r") as file:
            start_kp = json.load(file)
        with open(end_file, "r") as file:
            end_kp = json.load(file)
        
        start_2d = start_kp.get("2D_keypoint", [])
        end_2d = end_kp.get("2D_keypoint", [])
        start_dict = {point_pair[0]: point_pair[1] for point_pair in start_2d}
        end_dict = {point_pair[0]: point_pair[1] for point_pair in end_2d}
        common_indices = set(start_dict.keys()) & set(end_dict.keys())
        
        if not common_indices:
            return
        total_frames = end_frame - start_frame
        if total_frames <= 1:
            return
        for frame in range(start_frame + 1, end_frame):
            alpha = (frame - start_frame) / total_frames
            
            frame_file = f"{self.video_dir}/kp_record/{str(frame).zfill(5)}.json"
            if os.path.exists(frame_file):
                with open(frame_file, "r") as file:
                    current_kp = json.load(file)
            else:
                current_kp = {"2D_keypoint": []}
            interpolated_2d = []
            for obj_idx in common_indices:
                start_pos = start_dict[obj_idx]
                end_pos = end_dict[obj_idx]
                interp_x = start_pos[0] * (1 - alpha) + end_pos[0] * alpha
                interp_y = start_pos[1] * (1 - alpha) + end_pos[1] * alpha
                interpolated_2d.append([obj_idx, [interp_x, interp_y]])
            current_kp["2D_keypoint"] = interpolated_2d
            with open(frame_file, "w") as file:
                json.dump(current_kp, file, indent=4)
        print(f"show video between {start_frame} and {end_frame}")
    
    def show_video_interval(self, start_frame, end_frame, multiview_modify=False):
        
        self.render_status_label.config(text="Processing...")
        if not multiview_modify:
            self.body_params, self.hand_poses, self.R_finals[start_frame:end_frame], self.t_finals[start_frame:end_frame], optimized_params = kp_use(self.output, self.hand_poses, self.obj_orgs[start_frame:end_frame], 
                self.sampled_orgs[start_frame:end_frame], self.R_finals[start_frame:end_frame],
                    self.t_finals[start_frame:end_frame], self.human_part, self.K, 
                start_frame, end_frame, self.joint_to_optimize, self.video_dir, self.is_static_object.get())
            self.unwrapped_body_params()
            self.end_frame = end_frame
        else:
            self.body_params, self.hand_poses, self.R_finals[start_frame:end_frame], self.t_finals[start_frame:end_frame], optimized_params = kp_use_multiview(
                self.output,
                self.hand_poses,
                self.obj_orgs[start_frame:end_frame],
                self.sampled_orgs[start_frame:end_frame],
                self.R_finals[start_frame:end_frame],
                self.t_finals[start_frame:end_frame],
                self.human_part,
                self.K,
                start_frame,
                end_frame,
                self.joint_to_optimize,
                self.video_dir,
                self.is_static_object.get()
            )
            self.unwrapped_body_params()
        
        # 暂存优化参数
        if optimized_params:
            self.temp_optimized_params.append(optimized_params)
            print(f"✓ Optimization parameters for frames {start_frame}-{end_frame} stored temporarily")
        
        import threading
        def create_and_show_video():
            optimized_dir = f"{self.video_dir}/optimized_frames"
            if not os.path.exists(optimized_dir):
                return
            
            frame_files = []
            for frame_idx in range(0, end_frame, 2):
                frame_file = f"{optimized_dir}/frame_{str(frame_idx).zfill(4)}.png"
                frame_files.append(frame_file)
            output_video = f"{self.video_dir}/temp_optimized.mp4"
            first_frame = cv2.imread(frame_files[0])
            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, 18.0, (width, height))
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    out.write(frame)
            out.release()
            import time
            time.sleep(5)
            def play_video():
                if hasattr(self, 'video_cap_render'):
                    self.video_cap_render.release()
                self.video_cap_render = cv2.VideoCapture(output_video)
                self.render_status_label.place_forget()
                self.render_video_label.place(x=0, y=0)
                def update_frame():
                    if hasattr(self, 'video_cap_render') and self.video_cap_render.isOpened():
                        ret, frame = self.video_cap_render.read()
                        if ret:
                            frame = cv2.resize(frame, (480, 480))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_img = Image.fromarray(frame)
                            self.render_video_img = ImageTk.PhotoImage(frame_img)
                            self.render_video_label.configure(image=self.render_video_img)
                            self.root.after(67, update_frame)
                        else:
                            self.video_cap_render.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.root.after(67, update_frame)
                update_frame()
            self.root.after(0, play_video)
        threading.Thread(target=create_and_show_video, daemon=True).start()   
        for i in range(start_frame, end_frame + 1):
            self.rendered_frames.add(i)

    def on_close(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_cap_render'):
            self.video_cap_render.release()
        self.root.destroy()
        self.root.quit()
    def frame_change(self, new_frame):
        self.current_frame = new_frame
        is_rendered = self.current_frame in self.rendered_frames
        for btn in self.annotation_buttons:
            btn.config(state=tk.DISABLED if is_rendered else tk.NORMAL)
        self.re_annotate_button.config(state=tk.NORMAL if is_rendered else tk.DISABLED)

    def save_final_optimized_parameters(self):
        """整合并保存所有优化参数到JSON文件（使用transform_and_save方式）"""
        if not self.temp_optimized_params:
            print("No temporary optimization parameters found to save")
            return
        
        try:
            # 备份关键数据以防意外修改
            backup_body_params = deepcopy(self.body_params) if hasattr(self, 'body_params') else None
            backup_output = deepcopy(self.output) if hasattr(self, 'output') else None
            backup_object_poses = deepcopy(self.object_poses) if hasattr(self, 'object_poses') else None
            
            print("🔒 Created backup of critical data before parameter saving")
            
            # 创建保存目录
            save_dir = os.path.join(self.video_dir, "final_optimized_parameters")
            os.makedirs(save_dir, exist_ok=True)
            
            # 初始化整合的参数字典
            final_human_params = {
                'body_pose': {},
                'betas': {},
                'global_orient': {},
                'transl': {},
                'left_hand_pose': {},
                'right_hand_pose': {},
            }
            
            final_object_params = {
                'poses': {},  # R_final
                'centers': {},  # t_final
                'scale': self.object_poses['scale']  # 添加用户设置的scale参数
            }
            
            # 整合所有段的参数
            tmp_copy=deepcopy(self.temp_optimized_params)
            for segment_params in tmp_copy:
                human_params = segment_params['human_params']
                object_params = segment_params['object_params']
                start_frame = segment_params['frame_range']['start_frame']
                
                # 将每一帧的参数添加到最终字典中
                for i, (body_pose, betas, global_orient, transl, left_hand, right_hand) in enumerate(zip(
                    human_params['body_pose'],
                    human_params['betas'],
                    human_params['global_orient'],
                    human_params['transl'],
                    human_params['left_hand_pose'],
                    human_params['right_hand_pose']
                )):
                    frame_idx = start_frame + i
                    final_human_params['body_pose'][str(frame_idx)] = body_pose
                    final_human_params['betas'][str(frame_idx)] = betas
                    final_human_params['global_orient'][str(frame_idx)] = global_orient
                    final_human_params['transl'][str(frame_idx)] = transl
                    final_human_params['left_hand_pose'][str(frame_idx)] = left_hand
                    final_human_params['right_hand_pose'][str(frame_idx)] = right_hand
                
                # 物体参数
                for i, (poses, centers) in enumerate(zip(
                    object_params['poses'],
                    object_params['centers']
                )):
                    frame_idx = start_frame + i
                    final_object_params['poses'][str(frame_idx)] = poses
                    final_object_params['centers'][str(frame_idx)] = centers
            
            # 获取用户设置的scale
            user_scale = float(self.scale_var.get())
            
            # 获取原始物体mesh路径
            original_object_path = os.path.join(self.video_dir, "obj_org.obj")
            
            # 使用新的参数变换和保存函数（使用深拷贝避免修改原始数据）
            print("🔄 Using transform_and_save approach for parameter saving...")
            
            # 创建完全独立的副本来避免修改原始数据
            safe_human_params = deepcopy(final_human_params)
            safe_object_params = deepcopy(final_object_params)
            safe_camera_params = deepcopy(self.output) if hasattr(self, 'output') else {}
            
            saved_files = transform_and_save_parameters(
                human_params_dict=safe_human_params,
                org_params=safe_object_params,  # 使用深拷贝的副本
                camera_params=safe_camera_params,     # 使用深拷贝的副本
                output_dir=save_dir,
                original_object_path=original_object_path,  # 添加原始物体路径
                user_scale=user_scale
            )
            
            # # 验证关键数据是否被意外修改，如果是则恢复备份
            # if backup_body_params is not None and hasattr(self, 'body_params'):
            #     if not self._data_unchanged(self.body_params, backup_body_params):
            #         print("⚠️ Warning: body_params was modified during save, restoring from backup")
            #         self.body_params = backup_body_params
            #         self.unwrapped_body_params()  # 重新处理参数
            
            # if backup_output is not None and hasattr(self, 'output'):
            #     if not self._data_unchanged(self.output, backup_output):
            #         print("⚠️ Warning: output was modified during save, restoring from backup")
            #         self.output = backup_output
            
            # if backup_object_poses is not None and hasattr(self, 'object_poses'):
            #     if not self._data_unchanged(self.object_poses, backup_object_poses):
            #         print("⚠️ Warning: object_poses was modified during save, restoring from backup")
            #         self.object_poses = backup_object_poses
            
            # print("✅ Data integrity verification completed")
            
            # 在UI中显示保存信息  
            total_segments = len(self.temp_optimized_params)
            current_scale = float(self.scale_var.get())
            self.point_info.insert(tk.END, f"🎉 Parameters saved successfully!\n")
            self.point_info.insert(tk.END, f"📁 Location: {save_dir}\n")
            self.point_info.insert(tk.END, f"📊 Total frames: {len(final_human_params['body_pose'])}\n")
            self.point_info.insert(tk.END, f"📊 Total segments: {total_segments}\n")
            self.point_info.insert(tk.END, f"⚖️ Object scale: {current_scale}x\n")
            self.point_info.insert(tk.END, f"📄 Files saved: {len(saved_files)}\n")
            self.point_info.insert(tk.END, "✅ All transforms pre-applied to parameters\n")
            if any(f.endswith('.obj') for f in saved_files):
                self.point_info.insert(tk.END, "🎯 Transformed object mesh saved\n")
            self.point_info.insert(tk.END, "=" * 40 + "\n")
            self.point_info.see(tk.END)
            
            # 清空暂存参数
            # self.temp_optimized_params = []
            
        except Exception as e:
            # 如果出现异常，恢复所有备份数据
            if 'backup_body_params' in locals() and backup_body_params is not None:
                self.body_params = backup_body_params
                print("🔄 Restored body_params from backup due to exception")
            
            if 'backup_output' in locals() and backup_output is not None:
                self.output = backup_output
                print("🔄 Restored output from backup due to exception")
                
            if 'backup_object_poses' in locals() and backup_object_poses is not None:
                self.object_poses = backup_object_poses
                print("🔄 Restored object_poses from backup due to exception")
            
            print(f"Error saving final parameters: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("保存错误", f"保存最终参数时出错: {str(e)}")
    
    def _data_unchanged(self, data1, data2):
        """检查两个数据结构是否相同（简单版本）"""
        try:
            # 简单的类型和结构检查
            if type(data1) != type(data2):
                return False
            
            if isinstance(data1, dict):
                if set(data1.keys()) != set(data2.keys()):
                    return False
                # 只检查结构，不深入比较值（避免性能问题）
                return True
            elif hasattr(data1, 'shape'):  # numpy/torch数组
                if hasattr(data2, 'shape'):
                    return data1.shape == data2.shape
            
            return True  # 其他情况假设未改变
        except Exception:
            return True  # 如果检查失败，假设未改变

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Body-Object关键点标注工具")
    parser.add_argument('--video_dir', type=str, required=True, help='包含对象的基础目录')
    parser.add_argument('--joint_to_optimize', type=str, required=True, help='要优化的关节')
    args = parser.parse_args()
    root = ttk_boot.Window(themename="superhero")  # 使用ttkbootstrap的Window
    app = KeyPointApp(root, args)
    root.mainloop()

