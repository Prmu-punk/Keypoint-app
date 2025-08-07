# Runtime hook for PIL and Tkinter integration in PyInstaller
# This hook ensures proper initialization of PIL's tkinter interface

import sys
import os

# 确保 PIL._tkinter_finder 模块可以正确导入
try:
    import PIL._tkinter_finder
except ImportError:
    # 如果无法导入，尝试手动设置
    pass

# 确保 PIL._imagingtk 模块可以正确导入
try:
    import PIL._imagingtk
except ImportError:
    pass

# 确保 tkinter 模块正确初始化
try:
    import tkinter
    # 强制初始化 tkinter
    root = tkinter.Tk()
    root.withdraw()  # 隐藏主窗口
    root.destroy()   # 销毁窗口
except Exception as e:
    print(f"Warning: Tkinter initialization failed: {e}")

# 设置环境变量以帮助 PIL 找到 tkinter
if hasattr(sys, '_MEIPASS'):
    # 在 PyInstaller 环境中
    meipass = sys._MEIPASS
    
    # 添加 PIL 相关路径到 sys.path
    pil_path = os.path.join(meipass, 'PIL')
    if os.path.exists(pil_path) and pil_path not in sys.path:
        sys.path.insert(0, pil_path)
    
    # 添加 tkinter 相关路径
    tk_path = os.path.join(meipass, 'tkinter')
    if os.path.exists(tk_path) and tk_path not in sys.path:
        sys.path.insert(0, tk_path)