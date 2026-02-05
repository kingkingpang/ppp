"""
流程串联模块 (Pipeline)

维护图像处理流水线，支持将多个算法按顺序链接，并记录每一步的中间结果。
"""

import numpy as np
from typing import List, Tuple, Callable, Any, Optional


class ImageLab:
    """
    图像处理实验室类
    
    核心流水线管理类，维护处理步骤的中间结果，支持算法链的串联执行。
    """
    
    def __init__(self):
        """
        初始化ImageLab实例
        
        pipeline_data格式: [(step_name, img), ...]
        每个元素是一个元组，包含步骤名称和对应的图片数组
        """
        self.pipeline_data: List[Tuple[str, np.ndarray]] = []
    
    def record(self, name: str, img: np.ndarray) -> None:
        """
        记录流水线中的中间结果
        
        Args:
            name: 步骤名称（用于标识和可视化）
            img: 该步骤处理后的图片数组
        """
        if img is None:
            raise ValueError(f"无法记录空图片: {name}")
        self.pipeline_data.append((name, img))
    
    def run_pipeline(self, img: np.ndarray, funcs: List[Tuple[str, Callable, dict]]) -> None:
        """
        执行算法链流水线
        
        按照funcs中定义的顺序依次执行各个算法函数，并记录每一步的结果。
        原图会自动作为第一步记录为"Original"。
        
        Args:
            img: 输入的原始图片
            funcs: 算法函数列表，格式为 [(name, func, params), ...]
                   - name: 步骤名称
                   - func: 算法函数（第一个参数必须是img）
                   - params: 算法函数的参数字典（kwargs）
        
        Example:
            >>> lab = ImageLab()
            >>> funcs = [
            ...     ("Grayscale", grayscale_func, {}),
            ...     ("Blur", blur_func, {"ksize": 5})
            ... ]
            >>> lab.run_pipeline(img, funcs)
        """
        # 清空之前的数据
        self.pipeline_data = []
        
        # 记录原始图片
        self.record("Original", img)
        
        # 依次执行各个算法函数
        current = img.copy()
        for name, func, params in funcs:
            try:
                # 调用算法函数，第一个参数是img，其余参数通过**params传递
                current = func(current, **params)
                self.record(name, current)
            except Exception as e:
                raise RuntimeError(f"执行步骤 '{name}' 时出错: {e}")
    
    def clear(self) -> None:
        """
        清空流水线数据
        
        重置pipeline_data为空列表，释放内存。
        """
        self.pipeline_data = []
    
    def get_results(self) -> List[Tuple[str, np.ndarray]]:
        """
        获取当前流水线的所有结果
        
        Returns:
            List[Tuple[str, np.ndarray]]: 格式为 [(step_name, img), ...] 的结果列表
        """
        return self.pipeline_data.copy()
    
    def get_step(self, name: str) -> Optional[np.ndarray]:
        """
        根据步骤名称获取对应的图片
        
        Args:
            name: 步骤名称
            
        Returns:
            numpy.ndarray: 对应的图片数组，如果不存在返回None
        """
        for step_name, img in self.pipeline_data:
            if step_name == name:
                return img
        return None
