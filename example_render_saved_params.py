#!/usr/bin/env python3
"""
render_saved_params.py 使用示例
展示如何使用Renderer渲染保存的参数
"""

import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_usage():
    """示例用法"""
    
    # 示例文件路径（请根据实际情况修改）
    base_dir = r"c:\Users\wbr20\PROJECTS\4dhoi\keypoints"
    
    # 输入文件路径
    transformed_params_file = os.path.join(base_dir, "kp_app_design", "Keypoint-app", "final_optimized_parameters", "transformed_parameters_20250803_120000.json")
    transformed_object_file = os.path.join(base_dir, "kp_app_design", "Keypoint-app", "final_optimized_parameters", "transformed_object.obj")
    
    # 输出视频路径
    output_dir = os.path.join(base_dir, "kp_app_design", "visualization", "rendered_videos")
    human_only_video = os.path.join(output_dir, "human_global_view.mp4")
    combined_video = os.path.join(output_dir, "combined_global_view.mp4")
    
    print("🎬 Renderer-based Parameter Visualization Example")
    print("=" * 60)
    
    # 检查输入文件
    print("📋 Checking input files:")
    files_to_check = [
        ("Transformed parameters", transformed_params_file),
        ("Transformed object", transformed_object_file)
    ]
    
    for name, filepath in files_to_check:
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {filepath}")
        if not exists and name == "Transformed parameters":
            print(f"     Error: Required file not found!")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # 示例1: 仅渲染人体
    print("\n🎬 Example 1: Human-only global view rendering")
    print("-" * 40)
    
    cmd1 = f'''python render_saved_params.py \\
    --transformed_params "{transformed_params_file}" \\
    --output_video "{human_only_video}" \\
    --width 1024 \\
    --height 1024 \\
    --fps 30'''
    
    print("Command:")
    print(cmd1)
    print("\nThis will create a global view video showing only the human body.")
    
    # 示例2: 渲染人体+物体组合
    if os.path.exists(transformed_object_file):
        print("\n🎬 Example 2: Combined human+object view rendering")
        print("-" * 50)
        
        cmd2 = f'''python render_saved_params.py \\
    --transformed_params "{transformed_params_file}" \\
    --transformed_object "{transformed_object_file}" \\
    --output_video "{combined_video}" \\
    --width 1024 \\
    --height 1024 \\
    --fps 30 \\
    --combined'''
        
        print("Command:")
        print(cmd2)
        print("\nThis will create a global view video showing both human and object.")
    
    # 参数说明
    print("\n📖 Parameter explanations:")
    print("-" * 30)
    param_explanations = [
        ("--transformed_params", "Path to transformed parameters JSON file (required)"),
        ("--transformed_object", "Path to transformed object mesh file (optional)"),
        ("--output_video", "Output video file path (required)"),
        ("--width", "Video width in pixels (default: 1024)"),
        ("--height", "Video height in pixels (default: 1024)"),
        ("--fps", "Video frame rate (default: 30)"),
        ("--crf", "Video compression quality, lower = better quality (default: 18)"),
        ("--combined", "Enable combined human+object rendering (flag)"),
        ("--smpl_model", "Custom SMPL model path (optional, has default)")
    ]
    
    for param, desc in param_explanations:
        print(f"  {param:<20} : {desc}")
    
    # 渲染特性说明
    print("\n🎯 Rendering features:")
    print("-" * 25)
    features = [
        "✅ Global camera view with automatic camera positioning",
        "✅ Ground plane for better spatial reference", 
        "✅ High-quality mesh rendering with lighting",
        "✅ Smooth camera movement around the scene",
        "✅ Support for human-only or combined human+object rendering",
        "✅ Customizable video resolution and quality",
        "✅ Uses transformed parameters directly (no re-computation)"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    # 技术细节
    print("\n🔧 Technical details:")
    print("-" * 25)
    tech_details = [
        "🎥 Uses get_global_cameras_static for automatic camera path generation",
        "🎨 Uses Renderer class for high-quality mesh rendering",
        "🌍 Automatically computes ground plane from scene geometry", 
        "💡 Includes proper lighting setup for realistic appearance",
        "📐 Supports custom camera intrinsics and extrinsics",
        "🎬 Outputs standard MP4 video format"
    ]
    
    for detail in tech_details:
        print(f"  {detail}")
    
    print("\n" + "=" * 60)
    print("🚀 Ready to render! Run the commands above to generate videos.")
    print("💡 Tip: Start with human-only rendering to test, then add objects.")

if __name__ == "__main__":
    example_usage()
