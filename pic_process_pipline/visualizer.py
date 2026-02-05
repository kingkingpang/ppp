"""
可视化模块 (Visualizer)

提供统一的绘图接口，支持矩阵化可视化展示。
自动计算子图布局，支持横向对比（不同算法）和纵向对比（同算法不同参数）。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import math


def show_results(results: List[Tuple[str, np.ndarray]], title: str = "Pipeline Contrast", 
                 figsize: Optional[Tuple[int, int]] = None) -> None:
    """
    展示处理结果矩阵
    
    根据结果数量自动计算子图布局，统一处理灰度图和彩色图的显示。
    
    Args:
        results: 结果列表，格式为 [(step_name, img), ...]
        title: 整个图的标题
        figsize: 图片大小 (width, height)，如果为None则自动计算
        
    Example:
        >>> results = [
        ...     ("Original", img1),
        ...     ("Filtered", img2),
        ...     ("Enhanced", img3)
        ... ]
        >>> show_results(results, title="处理流程对比")
    """
    if not results:
        print("警告: 结果列表为空，无法显示")
        return
    
    n = len(results)
    
    # 自动计算布局：列数 = ceil(sqrt(n))，行数自适应
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # 如果没有指定figsize，根据行列数自动计算
    if figsize is None:
        # 每个子图约4x4英寸，加上边距
        figsize = (cols * 4, rows * 4)
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 处理单行或单列的情况
    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1) if isinstance(axes, np.ndarray) else [[ax] for ax in axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    # 显示每个结果
    for idx, (step_name, img) in enumerate(results):
        if idx >= len(axes):
            break
            
        ax = axes[idx] if rows == 1 or cols == 1 else axes[idx]
        
        # 检测是否为灰度图：维度为2或通道数为1
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        
        # 如果是3维但通道数为1，需要压缩维度
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)
        
        # 显示图片
        if is_grayscale:
            ax.imshow(img, cmap='gray')
        else:
            # OpenCV读取的是BGR格式，需要转换为RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = img[:, :, ::-1]  # BGR -> RGB
                ax.imshow(img_rgb)
            else:
                ax.imshow(img)
        
        # 设置标题和关闭坐标轴
        ax.set_title(step_name, fontsize=12)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        if rows == 1 or cols == 1:
            axes[idx].axis('off')
        else:
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_grid(results: List[Tuple[str, np.ndarray]], rows: int, cols: int, 
              title: str = "Grid Comparison", figsize: Optional[Tuple[int, int]] = None) -> None:
    """
    以指定行列数显示结果网格
    
    用于需要精确控制布局的场景，如同算法不同参数的对比。
    
    Args:
        results: 结果列表，格式为 [(step_name, img), ...]
        rows: 行数
        cols: 列数
        title: 整个图的标题
        figsize: 图片大小 (width, height)，如果为None则自动计算
        
    Example:
        >>> results = [
        ...     ("Param1", img1),
        ...     ("Param2", img2),
        ...     ("Param3", img3),
        ...     ("Param4", img4)
        ... ]
        >>> show_grid(results, rows=2, cols=2, title="参数对比")
    """
    if not results:
        print("警告: 结果列表为空，无法显示")
        return
    
    n = len(results)
    total = rows * cols
    
    if n > total:
        print(f"警告: 结果数量({n})超过网格大小({total})，只显示前{total}个")
        results = results[:total]
    
    if figsize is None:
        figsize = (cols * 4, rows * 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 处理单行或单列的情况
    if rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1) if isinstance(axes, np.ndarray) else [[ax] for ax in axes]
    else:
        axes = axes.flatten()
    
    for idx, (step_name, img) in enumerate(results):
        ax = axes[idx]
        
        # 检测是否为灰度图
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(axis=2)
        
        if is_grayscale:
            ax.imshow(img, cmap='gray')
        else:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = img[:, :, ::-1]  # BGR -> RGB
                ax.imshow(img_rgb)
            else:
                ax.imshow(img)
        
        ax.set_title(step_name, fontsize=12)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
