"""
算法处理模块 (Processors)

存放所有图像处理算法的原子函数。
每个函数必须遵循以下接口规范：
- 第一个参数必须是 img (numpy.ndarray)
- 返回处理后的图片 (numpy.ndarray)
- 包含完整的 Docstring 说明参数含义
"""
import cv2
import numpy as np
from typing import Any


def example_processor(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    示例处理函数模板
    
    这是一个占位函数，展示如何添加新的算法函数。
    实际使用时，请根据具体算法需求实现相应的处理逻辑。
    
    Args:
        img: 输入图片数组 (numpy.ndarray)
        **kwargs: 其他算法特定参数
        
    Returns:
        numpy.ndarray: 处理后的图片数组
        
    Example:
        >>> processed = example_processor(img, param1=5, param2=10)
    """
    # 示例：直接返回原图（实际使用时替换为具体算法）
    return img.copy()


# ============================================================================
# 在此处添加您的算法函数
# ============================================================================
# 
# 添加新算法的模板：
# 
# def your_algorithm(img: np.ndarray, param1: int = 5, param2: float = 1.0) -> np.ndarray:
#     """
#     算法描述
#     
#     Args:
#         img: 输入图片数组
#         param1: 参数1的说明
#         param2: 参数2的说明
#         
#     Returns:
#         numpy.ndarray: 处理后的图片
#     """
#     # 算法实现
#     processed = ...  # 您的处理逻辑
#     return processed
# 
# ============================================================================

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    将彩色图像转换为灰度图像
    
    Args:
        img: 输入图像数组
        
    Returns:
        numpy.ndarray: 灰度图像
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    手动实现RGB到灰度的转换（加权平均法）
    
    使用标准权重: R*0.299 + G*0.587 + B*0.114
    
    Args:
        img: 输入RGB图像
        
    Returns:
        numpy.ndarray: 灰度图像
    """
    if len(img.shape) == 3:
        # 手动计算灰度值
        gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        return gray.astype(np.uint8)
    return img.copy()


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """
    高斯模糊滤波
    
    Args:
        img: 输入图像
        ksize: 核大小（必须为奇数）
        sigma: 高斯核标准差，0表示自动计算
        
    Returns:
        numpy.ndarray: 模糊后的图像
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    中值滤波（对椒盐噪声有效）
    
    Args:
        img: 输入图像
        ksize: 核大小（必须为奇数）
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    双边滤波（保持边缘的同时去噪）
    
    Args:
        img: 输入图像
        d: 滤波时周围每个像素领域的直径
        sigma_color: 颜色空间的标准方差
        sigma_space: 坐标空间的标准方差
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def mean_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    均值滤波
    
    Args:
        img: 输入图像
        ksize: 核大小
        
    Returns:
        numpy.ndarray: 滤波后的图像
    """
    return cv2.blur(img, (ksize, ksize))


def canny_edge_detection(img: np.ndarray, min_threshold: int = 100, max_threshold: int = 200) -> np.ndarray:
    """
    Canny边缘检测算法
    
    Args:
        img: 输入灰度图像
        min_threshold: 最小阈值
        max_threshold: 最大阈值
        
    Returns:
        numpy.ndarray: 二值边缘图像
    """
    return cv2.Canny(img, min_threshold, max_threshold)


def sobel_edge_detection(img: np.ndarray, dx: int = 1, dy: int = 0, ksize: int = 3) -> np.ndarray:
    """
    Sobel边缘检测
    
    Args:
        img: 输入灰度图像
        dx: x方向导数阶数
        dy: y方向导数阶数
        ksize: 核大小（1, 3, 5, 7）
        
    Returns:
        numpy.ndarray: 边缘梯度图像
    """
    return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)


def laplacian_edge_detection(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian边缘检测（二阶导数）
    
    Args:
        img: 输入灰度图像
        ksize: 核大小
        
    Returns:
        numpy.ndarray: 边缘图像
    """
    return cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)


def binary_threshold(img: np.ndarray, threshold: int = 127, max_value: int = 255, 
                    threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
    """
    全局阈值分割
    
    Args:
        img: 输入灰度图像
        threshold: 阈值
        max_value: 最大值
        threshold_type: 阈值类型（THRESH_BINARY, THRESH_BINARY_INV, etc.）
        
    Returns:
        numpy.ndarray: 二值图像
    """
    _, result = cv2.threshold(img, threshold, max_value, threshold_type)
    return result


def adaptive_threshold(img: np.ndarray, max_value: int = 255, 
                      adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      threshold_type: int = cv2.THRESH_BINARY, block_size: int = 11, 
                      C: int = 2) -> np.ndarray:
    """
    自适应阈值分割
    
    Args:
        img: 输入灰度图像
        max_value: 最大值
        adaptive_method: 自适应方法（ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C）
        threshold_type: 阈值类型
        block_size: 块大小（必须为奇数）
        C: 从平均值或加权平均值中减去的常数
        
    Returns:
        numpy.ndarray: 二值图像
    """
    return cv2.adaptiveThreshold(img, max_value, adaptive_method, threshold_type, block_size, C)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Otsu自动阈值分割
    
    Args:
        img: 输入灰度图像
        
    Returns:
        numpy.ndarray: 二值图像
    """
    _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def morphological_dilation(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    形态学膨胀操作
    
    Args:
        img: 输入二值图像
        kernel_size: 结构元素大小
        iterations: 迭代次数
        
    Returns:
        numpy.ndarray: 膨胀后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(img, kernel, iterations=iterations)


def morphological_erosion(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    形态学腐蚀操作
    
    Args:
        img: 输入二值图像
        kernel_size: 结构元素大小
        iterations: 迭代次数
        
    Returns:
        numpy.ndarray: 腐蚀后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=iterations)


def morphological_opening(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    形态学开运算（先腐蚀后膨胀，去除小物体）
    
    Args:
        img: 输入二值图像
        kernel_size: 结构元素大小
        iterations: 迭代次数
        
    Returns:
        numpy.ndarray: 开运算后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_closing(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    形态学闭运算（先膨胀后腐蚀，填充小孔）
    
    Args:
        img: 输入二值图像
        kernel_size: 结构元素大小
        iterations: 迭代次数
        
    Returns:
        numpy.ndarray: 闭运算后的图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_gradient(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    形态学梯度（膨胀-腐蚀，提取边缘）
    
    Args:
        img: 输入二值图像
        kernel_size: 结构元素大小
        
    Returns:
        numpy.ndarray: 形态学梯度图像
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def find_contours(img: np.ndarray, mode: int = cv2.RETR_EXTERNAL, 
                 method: int = cv2.CHAIN_APPROX_SIMPLE) -> tuple:
    """
    查找图像轮廓
    
    Args:
        img: 输入二值图像
        mode: 轮廓检索模式
        method: 轮廓近似方法
        
    Returns:
        tuple: (contours, hierarchy) - 轮廓列表和层级结构
    """
    return cv2.findContours(img, mode, method)


def draw_contours(img: np.ndarray, contours: list, contour_idx: int = -1, 
                 color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制轮廓
    
    Args:
        img: 输入图像（会被复制）
        contours: 轮廓列表
        contour_idx: 要绘制的轮廓索引，-1表示绘制所有轮廓
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        numpy.ndarray: 绘制了轮廓的图像
    """
    result = img.copy()
    cv2.drawContours(result, contours, contour_idx, color, thickness)
    return result


def filter_contours_by_area(contours: list, min_area: float = 0, max_area: float = float('inf')) -> list:
    """
    根据面积过滤轮廓
    
    Args:
        contours: 轮廓列表
        min_area: 最小面积
        max_area: 最大面积
        
    Returns:
        list: 过滤后的轮廓列表
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)
    return filtered



def resize_image(img: np.ndarray, width: int = None, height: int = None, 
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        img: 输入图像
        width: 目标宽度，如果为None则保持比例
        height: 目标高度，如果为None则保持比例
        interpolation: 插值方法
        
    Returns:
        numpy.ndarray: 调整大小后的图像
    """
    h, w = img.shape[:2]
    
    if width is None and height is None:
        return img.copy()
    
    if width is None:
        aspect_ratio = height / h
        width = int(w * aspect_ratio)
    elif height is None:
        aspect_ratio = width / w
        height = int(h * aspect_ratio)
    
    return cv2.resize(img, (width, height), interpolation=interpolation)


def rotate_image(img: np.ndarray, angle: float, center: tuple = None, 
                scale: float = 1.0) -> np.ndarray:
    """
    旋转图像
    
    Args:
        img: 输入图像
        angle: 旋转角度（度）
        center: 旋转中心点，如果为None则使用图像中心
        scale: 缩放比例
        
    Returns:
        numpy.ndarray: 旋转后的图像
    """
    h, w = img.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, rotation_matrix, (w, h))


def connectedComponentsWithStats(img: np.ndarray, connectivity: int = 8,
                               ltype: int = cv2.CV_32S) -> tuple:
    """
    连通组件分析（带统计信息）

    分析二值图像中的连通组件，返回标签图像和统计信息。
    常用于分析二值图像中的独立区域，如目标检测、分割等。

    Args:
        img: 输入二值图像（0和非0值）
        connectivity: 连通性，4或8，默认8（8连通）
        ltype: 输出标签图像的数据类型，默认cv2.CV_32S

    Returns:
        tuple: (labels, stats, centroids)
        - labels: 标签图像，每个像素值为其所属连通组件的标签
        - stats: 统计信息数组，每行对应一个连通组件的统计数据
          stats[i, cv2.CC_STAT_LEFT/TOP/WIDTH/HEIGHT/AREA] 分别为：
          左上角x坐标、左上角y坐标、宽度、高度、像素面积
        - centroids: 质心坐标数组，每个连通组件的质心(x,y)坐标

    Example:
        >>> labels, stats, centroids = connectedComponentsWithStats(binary_img)
        >>> print(f"找到 {stats.shape[0]} 个连通组件")
        >>> # 显示标签图像
        >>> show_results([("Labels", labels, {})])
    """
    # 确保输入是二值图像
    if len(img.shape) > 2:
        raise ValueError("输入图像必须是灰度或二值图像")

    # 调用OpenCV函数
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity, ltype
    )

    return labels, stats, centroids


def filter_connected_components(img: np.ndarray,
                              lab=None,
                              filter_type: str = "most_circular",
                              top_n: int = 5,
                              area_similarity_threshold: float = 0.3) -> np.ndarray:
    """
    连通组件筛选函数

    使用之前保存的连通域信息进行筛选，避免重复计算

    Args:
        img: 彩色标签图像（来自analyze_connected_components的输出）
        lab: ImageLab实例，用于获取保存的连通域信息
        filter_type: 筛选类型
            - "most_circular": 筛选最圆的组件
            - "area_similar": 筛选面积相似的组件
            - "combined": 结合两种筛选条件
        top_n: 选择前N个最圆的组件（用于most_circular模式）
        area_similarity_threshold: 面积相似性阈值，越小要求越相似（用于area_similar模式）

    Returns:
        彩色标签图像，只显示筛选出的组件，背景为黑色
    """
    try:
        # 获取之前保存的连通域信息
        if lab is None:
            raise ValueError("需要提供ImageLab实例来访问连通域信息")

        labels = lab.get_intermediate_result('cc_labels')
        stats = lab.get_intermediate_result('cc_stats')
        centroids = lab.get_intermediate_result('cc_centroids')
        binary_img = lab.get_intermediate_result('cc_binary_img')

        if labels is None or stats is None or binary_img is None:
            print("  没有找到连通域信息")
            return img

        num_components = stats.shape[0]

        if num_components <= 1:
            print("  没有找到有效的连通组件")
            return np.zeros((*img.shape[:2], 3), dtype=np.uint8)

        # 计算每个组件的圆度和面积
        component_info = []

        for i in range(1, num_components):  # 跳过背景(索引0)
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # 创建单个组件的掩码
            component_mask = (labels == i).astype(np.uint8) * 255

            # 查找轮廓（使用保存的二值图像）
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 计算周长
                perimeter = cv2.arcLength(contours[0], True)

                # 计算圆度 (4π * 面积 / 周长²)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                else:
                    circularity = 0

                component_info.append({
                    'index': i,
                    'area': area,
                    'circularity': circularity,
                    'perimeter': perimeter,
                    'bbox': (left, top, width, height),
                    'centroid': centroids[i]
                })

        if not component_info:
            print("  无法计算组件特征")
            return np.zeros((*img.shape[:2], 3), dtype=np.uint8)

        # 根据筛选类型选择组件
        selected_indices = []

        if filter_type == "most_circular":
            # 按圆度排序，选择前top_n个
            sorted_by_circularity = sorted(component_info,
                                         key=lambda x: x['circularity'],
                                         reverse=True)
            selected_info = sorted_by_circularity[:top_n]
            selected_indices = [info['index'] for info in selected_info]

            print(f"  筛选出最圆的 {len(selected_indices)} 个组件:")
            for i, info in enumerate(selected_info):
                print(f"    #{i+1}: 圆度={info['circularity']:.3f}, 面积={info['area']}")

        elif filter_type == "area_similar":
            # 计算面积统计
            areas = [info['area'] for info in component_info]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv = std_area / mean_area if mean_area > 0 else 0

            print(f"  面积统计: 平均={mean_area:.1f}, 标准差={std_area:.1f}, CV={cv:.3f}")

            # 筛选面积相似的组件
            selected_info = []
            for info in component_info:
                area_diff = abs(info['area'] - mean_area) / mean_area
                if area_diff <= area_similarity_threshold:
                    selected_info.append(info)

            selected_indices = [info['index'] for info in selected_info]

            print(f"  筛选出面积相似的 {len(selected_indices)} 个组件 (阈值={area_similarity_threshold}):")
            for info in selected_info:
                area_diff = abs(info['area'] - mean_area) / mean_area
                print(f"    组件{info['index']}: 面积={info['area']}, 差异={area_diff:.3f}")

        elif filter_type == "combined":
            # 结合圆度和面积相似性
            areas = [info['area'] for info in component_info]
            mean_area = np.mean(areas)

            # 首先筛选面积相似的，再选择其中最圆的
            similar_area_info = []
            for info in component_info:
                area_diff = abs(info['area'] - mean_area) / mean_area
                if area_diff <= area_similarity_threshold:
                    similar_area_info.append(info)

            if similar_area_info:
                sorted_by_circularity = sorted(similar_area_info,
                                             key=lambda x: x['circularity'],
                                             reverse=True)
                selected_info = sorted_by_circularity[:top_n]
                selected_indices = [info['index'] for info in selected_info]

                print(f"  结合筛选: 面积相似且最圆的 {len(selected_indices)} 个组件")
                for info in selected_info:
                    print(f"    组件{info['index']}: 圆度={info['circularity']:.3f}, 面积={info['area']}")
            else:
                print("  没有找到面积相似的组件")
                selected_indices = []

        # 创建筛选结果的可视化图像
        if selected_indices:
            # 创建新的彩色图像，背景为黑色
            filtered_result = np.zeros((*labels.shape, 3), dtype=np.uint8)

            # 设置随机种子保证颜色一致性
            np.random.seed(42)
            colors = np.random.randint(50, 255, (len(selected_indices), 3), dtype=np.uint8)

            for i, comp_idx in enumerate(selected_indices):
                filtered_result[labels == comp_idx] = colors[i]

            return filtered_result
        else:
            # 没有筛选出组件，返回全黑图像
            return np.zeros((*img.shape[:2], 3), dtype=np.uint8)

    except Exception as e:
        print(f"  连通组件筛选失败: {e}")
        return img.copy()
