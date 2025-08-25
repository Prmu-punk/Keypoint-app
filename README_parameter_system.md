# 参数保存与可视化系统使用指南

## 系统概述

这个系统提供了完整的参数保存和高质量可视化流程：

1. **参数保存**: 在优化完成后自动保存转换后的参数和物体mesh
2. **高质量渲染**: 使用Renderer系统生成全局视角的高质量视频

## 文件结构

```
Keypoint-app/
├── app_main.py                        # 主应用程序（包含参数保存功能）
├── render_saved_params.py             # 新的Renderer渲染系统
├── example_render_saved_params.py     # 使用示例
├── README_parameter_system.md         # 本文件
├── final_optimized_parameters/        # 保存的参数目录
│   ├── transformed_parameters_*.json  # 转换后的参数文件
│   └── transformed_object.obj         # 转换后的物体mesh
└── video_optimizer/
    └── utils/
        └── parameter_transform.py     # 参数转换工具模块
```

## 使用流程

### 第1步：参数优化与保存

运行主应用程序进行参数优化：

```bash
python app_main.py
```

优化完成后，系统会自动：
- ✅ 转换SMPL参数到最终坐标系
- ✅ 应用尺度变换到物体mesh
- ✅ 保存到 `final_optimized_parameters/` 目录

### 第2步：高质量可视化

使用新的Renderer系统生成视频：

```bash
# 仅渲染人体（全局视角）
python render_saved_params.py \
    --transformed_params "final_optimized_parameters/transformed_parameters_20250803_120000.json" \
    --output_video "human_global_view.mp4" \
    --width 1024 --height 1024 --fps 30

# 渲染人体+物体组合（全局视角）
python render_saved_params.py \
    --transformed_params "final_optimized_parameters/transformed_parameters_20250803_120000.json" \
    --transformed_object "final_optimized_parameters/transformed_object.obj" \
    --output_video "combined_global_view.mp4" \
    --width 1024 --height 1024 --fps 30 --combined
```

## 参数文件格式

### transformed_parameters_*.json 结构：

```json
{
    "metadata": {
        "timestamp": "2025-08-03 12:00:00",
        "total_frames": 76,
        "transform_applied": {"R_final": "3x3 matrix", "t_final": "3x1 vector"}
    },
    "transformed_smpl_params": {
        "frame_000": {
            "body_pose": [72-dim array],
            "global_orient": [3-dim array], 
            "transl": [3-dim array],
            "betas": [10-dim array]
        },
        // ... more frames
    }
}
```

### transformed_object.obj 格式：

标准OBJ格式的物体mesh文件，已应用以下变换：
- ✅ 应用用户输入的scale参数  
- ✅ 应用R_final和t_final变换
- ✅ 与人体坐标系对齐

## 渲染系统特性

### 🎥 全局相机系统
- 使用 `get_global_cameras_static()` 自动生成相机路径
- 相机高度角度: 20°
- 目标中心高度: 1.0m
- 平滑的环绕运动

### 🌍 地面渲染
- 自动从场景几何计算地面参数
- 使用 `get_ground_params_from_points()`
- 地面尺度: object_scale * 1.5

### 💡 照明设置
- 颜色: `torch.ones(3).float().cuda() * 0.8` (柔和白光)
- 与全局相机系统集成
- 适合人体和物体的照明

### 🎬 视频输出
- 格式: MP4 (H.264编码)
- 默认参数: 1024x1024, 30fps, CRF=18
- 高质量压缩设置

## 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--transformed_params` | string | 必需 | 转换后参数JSON文件路径 |
| `--transformed_object` | string | 可选 | 转换后物体mesh文件路径 |
| `--output_video` | string | 必需 | 输出视频文件路径 |
| `--width` | int | 1024 | 视频宽度(像素) |
| `--height` | int | 1024 | 视频高度(像素) |
| `--fps` | int | 30 | 视频帧率 |
| `--crf` | int | 18 | 视频质量(越低质量越好) |
| `--combined` | flag | False | 启用人体+物体组合渲染 |
| `--smpl_model` | string | 默认路径 | 自定义SMPL模型路径 |

## 故障排除

### 常见问题

1. **找不到参数文件**
   ```
   Error: Transformed parameters file not found
   ```
   - 确保先运行 `app_main.py` 完成优化
   - 检查 `final_optimized_parameters/` 目录

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 降低视频分辨率: `--width 512 --height 512`
   - 重启Python释放GPU内存

3. **SMPL模型加载失败**
   ```
   Error: Could not load SMPL model
   ```
   - 检查 `SMPLX_NEUTRAL.npz` 文件是否存在
   - 使用 `--smpl_model` 指定正确路径

### 性能优化

- **快速预览**: 使用512x512分辨率和15fps
- **高质量输出**: 使用1920x1080分辨率和CRF=15
- **节省空间**: 使用CRF=23减小文件大小

## 示例脚本

运行示例脚本查看详细用法：

```bash
python example_render_saved_params.py
```

该脚本会：
- ✅ 检查必需文件是否存在
- ✅ 显示完整的命令行示例
- ✅ 解释所有参数和功能
- ✅ 提供故障排除建议

## 系统架构

```
app_main.py
    ↓ 优化完成
parameter_transform.py (utils)
    ↓ 转换参数
final_optimized_parameters/
    ↓ 加载参数
render_saved_params.py
    ↓ Renderer渲染
output_video.mp4
```

## 更新历史

- **v3.0**: 新增Renderer系统，支持全局相机和地面渲染
- **v2.0**: 参数转换移至utils模块，简化保存逻辑  
- **v1.0**: 基础参数保存和Open3D可视化

---

🎯 **快速开始**: 运行 `python example_render_saved_params.py` 查看完整示例！
