"""
使用示例脚本

展示如何使用图像处理实验框架进行基本的图像处理实验。
"""

import cv2
import numpy as np
from loaders import load_image
from processors import example_processor
from pipeline import ImageLab
from visualizer import show_results, show_grid


def simple_grayscale(img: np.ndarray) -> np.ndarray:
    """
    简单的灰度化处理函数示例
    
    Args:
        img: 输入图片数组
        
    Returns:
        numpy.ndarray: 灰度化后的图片
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def simple_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    简单的模糊处理函数示例
    
    Args:
        img: 输入图片数组
        ksize: 模糊核大小（必须是奇数）
        
    Returns:
        numpy.ndarray: 模糊处理后的图片
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def main():
    """
    主函数：演示基本使用流程
    """
    print("=" * 60)
    print("图像处理实验框架 - 使用示例")
    print("=" * 60)
    
    # 示例1: 单张图片处理
    print("\n示例1: 单张图片加载和处理")
    print("-" * 60)
    
    try:
        # 注意：这里使用示例路径，实际使用时请替换为您的图片路径
        # img_path = "test.jpg"  # 取消注释并替换为实际路径
        # img = load_image(img_path)
        
        # 为了演示，我们创建一个示例图片
        print("创建示例图片...")
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        print(f"图片形状: {img.shape}")
        
        # 创建ImageLab实例
        lab = ImageLab()
        
        # 定义处理步骤
        funcs = [
            ("Grayscale", simple_grayscale, {}),
            ("Blur", simple_blur, {"ksize": 5}),
        ]
        
        # 运行流水线
        print("运行处理流水线...")
        lab.run_pipeline(img, funcs)
        
        # 获取结果并可视化
        results = lab.get_results()
        print(f"流水线包含 {len(results)} 个步骤")
        print("步骤列表:", [name for name, _ in results])
        
        # 显示结果
        print("显示处理结果...")
        show_results(results, title="图像处理流水线示例")
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 如果您有实际的图片文件，可以取消注释代码中的图片加载部分")
    
    # 示例2: 参数扫描（Grid Search）
    print("\n\n示例2: 参数扫描对比")
    print("-" * 60)
    
    try:
        # 创建示例灰度图
        test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        # 测试不同模糊参数
        results = [("Original", test_img)]
        for ksize in [3, 5, 7, 9]:
            blurred = simple_blur(test_img, ksize=ksize)
            results.append((f"Blur ksize={ksize}", blurred))
        
        # 使用网格显示
        print("显示参数扫描结果（2x3网格）...")
        show_grid(results, rows=2, cols=3, title="模糊参数对比")
        
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 60)
    print("示例演示完成！")
    print("=" * 60)
    print("\n使用提示:")
    print("1. 在 processors.py 中添加您的算法函数")
    print("2. 使用 load_image() 加载单张图片")
    print("3. 使用 load_images() 批量处理文件夹中的图片（生成器模式）")
    print("4. 使用 ImageLab 类管理处理流水线")
    print("5. 使用 show_results() 或 show_grid() 可视化结果")


if __name__ == "__main__":
    main()
