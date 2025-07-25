
import tkinter as tk
import argparse
from tkinter import ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import numpy as np
import torch
import open3d as o3d
import os
import sys
import smplx
import trimesh
import shutil
import ttkbootstrap as ttk_boot
from ttkbootstrap.constants import *
import platform

from video_optimizer.utils.hoi_utils import load_transformation_matrix, update_hand_pose
from video_optimizer.kp_use import kp_use
from copy import deepcopy
from rotate_smpl import matrix_to_axis_angle
from icppnp import solve_weighted_priority
from pykalman import KalmanFilter


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
        self.last_frame = None
        self.last_frame_2D = None
        self.no_annot = True
        self.no_annote_2D = True
        self.selected_human_kp = None
        self.selected_2d_point = None
        self.is_static_object = tk.BooleanVar(value=False)

        self.load_config_files()
        self.setup_ui()
        self.load_data(args)
    
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

        buttons = [
            ("Reset keypoints", self.reset_keypoints, "outline-danger"),
            ("Select 3D point", self.open_o3d_viewer, "primary"),
            ("Select 2D point", self.keypoint_2D, "outline-info"),
            ("Manage keypoints", self.manage_existing_keypoints, "outline-warning")
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
        self.current_frame = 0
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
            self.original_height, self.original_width = frame.size
            self.original_max_dim = max(self.original_height, self.original_width)
            self.keypoint_window_size = min(self.original_max_dim, 800)
            standard_size = (480, 480)
            frame = frame.resize(standard_size, Image.Resampling.LANCZOS)
            self.img_width, self.img_height = frame.size
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label = ttk_boot.Label(self.video_display_frame, image=self.obj_img)
            self.video_label.pack(pady=10)
            self.root.geometry(f"{2*self.img_width + 360+560}x{self.img_height+500}")
            self.is_playing = False
            self.update_video_frame()

        output = torch.load(f"{args.video_dir}/motion/result.pt")
        print(output.keys())
        self.body_params = output["smpl_params_incam"]

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
        
        self.obj_orgs, self.centers = preprocess_obj(self.obj_org, self.object_poses, os.path.join(args.video_dir, 'orient/'), self.total_frames)
        self.sampled_orgs, _ = preprocess_obj(self.sampled_obj, self.object_poses, os.path.join(args.video_dir, 'orient/'), self.total_frames)
        
        if "rotation" not in self.object_poses:
            self.object_poses['rotation'] = []
            for frame in range(self.current_frame, self.total_frames):
                self.object_poses['rotation'].append(np.eye(3))
        
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

        self.setup_human_keypoints()
        self.setup_bottom_images()
        self.update_frame_counter()
    
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
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize((480, 480), Image.Resampling.LANCZOS)
            self.obj_img = ImageTk.PhotoImage(image=frame)
            self.video_label.configure(image=self.obj_img)
            self.update_frame_counter()
        
        self.frame_change(frame_no)
        self.is_playing = was_playing
    
    def update_frame_counter(self):
        """更新帧计数器"""
        self.frame_label.config(text=f"{self.current_frame}/{self.total_frames - 1}")
    
    def update_video_frame(self):
        """更新视频帧"""
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_change(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((480, 480), Image.Resampling.LANCZOS)
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
    
    def keypoint_2D(self):
        """选择2D关键点"""
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
            
        cv2.namedWindow("2D keypoint selection")
        
        clicked_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point[0] = (x, y)
                cv2.circle(display_img, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(display_img, f"({x}, {y})", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("2D keypoint selection", display_img)
        
        cv2.setMouseCallback("2D keypoint selection", mouse_callback)
        cv2.imshow("2D keypoint selection", display_img)
        
        while clicked_point[0] is None:
            cv2.waitKey(30)
            try:
                if cv2.getWindowProperty("2D keypoint selection", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        
        try:
            cv2.destroyWindow("2D keypoint selection")
        except cv2.error:
            pass
        
        if clicked_point[0] is not None:
            x, y = clicked_point[0]
            x /= scale
            y /= scale
            self.selected_2d_point = (x, y)
            self.update_plot()
    
    def manage_existing_keypoints(self):
        """管理现有关键点"""
        current_file = f"{self.video_dir}/kp_record/{str(self.current_frame).zfill(5)}.json"
        if not os.path.exists(current_file):
            messagebox.showinfo("Info", "No keypoint annotation")
            return
        
        with open(current_file, "r") as file:
            current_kp = json.load(file)
        
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
        if "2D_keypoint" in current_kp and current_kp["2D_keypoint"]:
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
            if key != "2D_keypoint" and isinstance(value, int):
                if not has_3d_keypoints:
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
        
        if not current_kp.get("2D_keypoint", []) and not has_3d_keypoints:
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
    
    def reset_keypoints(self):
        """重置关键点"""
        self.kp_pair = {"2D_keypoint": []}
        self.point_info.delete('1.0', tk.END)
        self.point_info.insert(tk.END, "Keypoints reset\n")
        self.point_label.config(text="No point selected")
        self.human_status_label.config(text="Click human keypoint to annotate")
    
    def open_o3d_viewer(self):
        """打开3D查看器"""
        vertices = self.obj_orgs[self.current_frame].vertices
        vertices = vertices + self.centers[self.current_frame]
        self.obj_orgs[self.current_frame].vertices = o3d.utility.Vector3dVector(vertices)

        mesh = self.obj_orgs[self.current_frame]
        mesh.compute_vertex_normals()
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

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        elif mesh.has_triangles() and mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals)
            colors = (normals + 1) / 2
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            vertices = np.asarray(mesh.vertices)
            min_bound = vertices.min(axis=0)
            max_bound = vertices.max(axis=0)
            norm_vertices = (vertices - min_bound) / (max_bound - min_bound)
            colors = np.zeros((len(vertices), 3))
            colors[:, 0] = norm_vertices[:, 0]
            colors[:, 1] = norm_vertices[:, 1]
            colors[:, 2] = norm_vertices[:, 2]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(output.vertices[0])

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
                self.obj_point = str(nearest_idx)
            else:
                self.obj_point = str(picked_idx)
            self.point_label.config(text=f"📍 Selected point: {self.obj_point}")
        else:
            self.obj_point = None
            self.point_label.config(text="📍 No point selected")
        vis.destroy_window()
    
    def update_plot(self):
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

            for frame in range(self.current_frame, self.total_frames):
                save_name = f"{frame}".zfill(5)
                with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
                    json.dump(self.kp_pair, file, indent=4)
            self.render_status_label.config(text="Processing...")
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
            with open(f"{self.video_dir}/kp_record/{save_name}.json", "w") as file:
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
            del current_kp["2D_keypoint"][index]
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
            
            messagebox.showinfo("Success", "2D keypoint modified")
    
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
    
    def show_video_interval(self, start_frame, end_frame):
        
        self.render_status_label.config(text="Processing...")
        kp_use(self.output, self.hand_poses, self.obj_orgs[start_frame:end_frame], 
               self.sampled_orgs[start_frame:end_frame], self.centers[start_frame:end_frame], self.human_part, self.K, 
               start_frame, end_frame, self.joint_to_optimize, self.video_dir, self.is_static_object.get())
        
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
    
    def frame_change(self, new_frame):
        if (self.last_frame is not None and not self.no_annot):
            start_frame = min(self.last_frame, self.current_frame)
            end_frame = max(self.last_frame, self.current_frame)
            self.show_video_interval(start_frame, end_frame)
            self.no_annot = True
        
        if (hasattr(self, 'last_frame_2D') and self.last_frame_2D is not None and not self.no_annote_2D):
            start_frame = min(self.last_frame_2D, self.current_frame)
            end_frame = max(self.last_frame_2D, self.current_frame)
            self.interpolate_2D(start_frame, end_frame)
            self.no_annote_2D = True
        
        self.current_frame = new_frame
    
    def on_close(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_cap_render'):
            self.video_cap_render.release()
        self.root.destroy()
        self.root.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Body-Object关键点标注工具")
    parser.add_argument('--video_dir', type=str, required=True, help='包含对象的基础目录')
    parser.add_argument('--joint_to_optimize', type=str, required=True, help='要优化的关节')
    args = parser.parse_args()
    root = ttk_boot.Window(themename="superhero")  # 使用ttkbootstrap的Window
    app = KeyPointApp(root, args)
    root.mainloop()


