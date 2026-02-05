"""
图片加载模块 (Loaders)

提供单张图片和批量图片的加载功能，支持生成器模式以节省内存。
"""

import cv2
import os
from pathlib import Path
from typing import Generator, Optional


def load_image(path: str) -> Optional:
    """
    加载单张图片
    
    Args:
        path: 图片文件路径
        
    Returns:
        numpy.ndarray: 图片数组，如果加载失败返回None
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"图片文件不存在: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图片文件，可能格式不支持: {path}")
    
    return img


def load_images(folder_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')) -> Generator:
    """
    批量加载文件夹中的图片（生成器模式）
    
    使用yield关键字实现内存友好的批量处理，适合处理大批量图片。
    
    Args:
        folder_path: 图片文件夹路径
        extensions: 支持的图片文件扩展名元组
        
    Yields:
        tuple: (文件名, 图片数组) 的元组
        
    Raises:
        FileNotFoundError: 文件夹不存在时抛出
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在或不是有效目录: {folder_path}")
    
    # 遍历文件夹中的所有图片文件
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    yield (file_path.name, img)
                else:
                    print(f"警告: 无法读取文件 {file_path.name}，跳过")
            except Exception as e:
                print(f"警告: 处理文件 {file_path.name} 时出错: {e}，跳过")
